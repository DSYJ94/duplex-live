"""Independently audit all20 paired native outputs after both actors exit.

Uses unchanged Whisper-turbo speech auditing, with original WAV word clocks and
silent failures retained. Successful execution never implies semantic approval.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from portable.context.native_development import atomic, PROOFS
from portable.context.native_gradient_followup import process_birth


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','paired','python','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--controller-pid',type=int,required=True)
    ap.add_argument('--controller-birth',type=int,required=True)
    ap.add_argument('--paired-source-commit',required=True)
    ap.add_argument('--deadline-utc',required=True)
    args=ap.parse_args()
    deadline=datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00')).timestamp()
    args.out.mkdir(exist_ok=False)
    lock=(args.paired/'speech-followup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(schema='dl-runpod-paired20-independent-speech-followup-v1',pid=os.getpid(),
               phase='waiting_for_complete_paired20',paired_source=args.paired_source_commit,
               controller_pid=args.controller_pid,controller_birth=args.controller_birth,
               deadline_utc=args.deadline_utc,complete20=False,behavior_approved=False,training_approved=False)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat();atomic(args.out/'controller.json',state)
    children=[]
    def stop(*_):
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda a,b:(stop(),exit(128+a)))
    save()
    try:
        if process_birth(args.controller_pid)!=args.controller_birth:
            raise RuntimeError('Exact paired native controller must be live at dispatch')
        while time.time()<deadline:
            current=json.loads((args.paired/'execution.json').read_text())
            if current['pid']!=args.controller_pid or current['source_commit']!=args.paired_source_commit:
                raise RuntimeError('Original paired controller identity changed')
            if current['phase']=='complete' and current['complete_paired20'] is True:
                if process_birth(args.controller_pid) is None:break
            elif process_birth(args.controller_pid)!=args.controller_birth:
                raise RuntimeError('Paired native execution ended without complete20')
            state['observed_paired_phase']=current['phase'];save();time.sleep(10)
        else:raise TimeoutError('Paired20 did not complete before deadline')
        for device,role in enumerate(PROOFS):
            actor=json.loads((args.paired/role/'execution.json').read_text())
            if not actor['complete10'] or not actor['parity']['passed'] or [r['index'] for r in actor['cases']]!=list(range(10)):
                raise RuntimeError('All ten actual native cases and output parity required')
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(device);env.pop('OPENAI_API_KEY',None)
            folders=[args.paired/role/('case-%02d'%i) for i in range(10)]
            cmd=[str(args.python),'-u','-m','evals.moshi_speech_audit','--runs',*map(str,folders),
                 '--device','cuda','--trim-leading-silence','--trim-trailing-silence',
                 '--keep-silent-failures','--out',str(args.out/role)]
            log=(args.out/(role+'.log')).open('x')
            child=subprocess.Popen(cmd,cwd=args.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close();children.append(child)
        state.update(phase='independent_speech20',children=[p.pid for p in children]);save()
        for child in children:
            child.wait(timeout=max(1,deadline-time.time()))
            if child.returncode:raise RuntimeError('Independent speech process failed: '+str(child.returncode))
        reports={}
        for role in PROOFS:
            report=args.out/role/'report.json';measured=json.loads(report.read_text())
            if len(measured['results'])!=10:raise RuntimeError('Independent speech audit omitted cases')
            for index,row in enumerate(measured['results']):
                folder=args.paired/role/('case-%02d'%index)
                if Path(row['run'])!=folder or row['wav_sha256']!=hashlib.sha256((folder/'model.wav').read_bytes()).hexdigest():
                    raise RuntimeError('Independent speech audit changed actual native audio')
            reports[role]=hashlib.sha256(report.read_bytes()).hexdigest()
        state.update(phase='complete_speech20_only',complete20=True,reports_sha256=reports,
                     semantic_quiet_pause_receipt_and_audio_review_still_required=True)
        save()
    except BaseException as error:
        stop();state.update(phase='failed',error_type=type(error).__name__,error=str(error));save();raise


if __name__=='__main__':main()
