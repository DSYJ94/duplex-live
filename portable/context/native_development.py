"""Complete paired native10 on two real GPUs, without provider credentials.

Ordinary conversations run in parallel. Actual-ASR conversations run serially
with the other physical GPU assigned to the isolated recognition subprocess.
Cold/reused PCM, token and native-input parity must pass for BOTH models first.
No optimizer, scoring answers, scheduled worker or subset-approval path exists.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/projects/u6si/duplex-live')
BASE = ROOT / 'models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309'
PLAN = 'evals/fixtures/native-v2-development-20261019.json'
PROOFS = {'baseline520': '7510331', 'candidate7590052': '7590052'}


def atomic(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def command(case, proof, out, asr_python):
    data = 'native-curriculum-v2-data-7541875' if case['source'] == 'native-v2' else 'recovery-data-7407554'
    result = ['--assets', str(ROOT/'runs/context-spanning-assets-7454205/manifest.json'),
              '--base', str(BASE), '--data', str(ROOT/'runs'/data),
              '--scenario', case['scenario'], '--expected-kind', case['kind'],
              '--private-value', case.get('value', 'red circle'),
              '--adapter-proof', str(ROOT/'runs'/('context-training-proof-'+proof)),
              '--voice', 'accepted-synthetic',
              '--voice-reference', str(ROOT/'runs/voice-quality-7414595/codec/reference.wav'),
              '--voice-provenance', str(ROOT/'runs/voice-quality-7414595/codec/report.json'),
              '--out', str(out), '--backend-mode', case['backend']]
    if case['backend'] == 'observe-only':
        result += ['--train-history']
    else:
        result += ['--asr-python', str(asr_python)]
    return result


def compare_native(left, right):
    import torch
    from evals.moshi_model_download import sha256
    def events(folder):
        return [json.loads(line)['text_token'] for line in (folder/'events.jsonl').read_text().splitlines()]
    a = torch.load(left/'student-history.pt', map_location='cpu', weights_only=True)
    b = torch.load(right/'student-history.pt', map_location='cpu', weights_only=True)
    checks = dict(packet_stream_wav_sha256_exact=sha256(left/'packet-stream.wav') == sha256(right/'packet-stream.wav'),
                  sampled_text_tokens_exact=events(left) == events(right),
                  actual_native_delayed_inputs_exact=torch.equal(a['delayed_inputs'], b['delayed_inputs']),
                  sampled_text_exact=json.loads((left/'report.json').read_text())['text'] ==
                                     json.loads((right/'report.json').read_text())['text'])
    if not all(checks.values()):
        raise RuntimeError('Native cold/reused session parity failed: '+repr(checks))
    return dict(checks, passed=True, intervening_different_conversation=True,
                actual_cuda_lm_and_pcm=True, wall_clock_playout_not_compared=True,
                behavior_or_training_approved=False)


def actor(args):
    # Set the main CUDA device before importing Torch. Every native-ASR call
    # receives BOTH actual devices; the unchanged probe isolates its main one.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    if os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('Provider credential must remain on Mac')
    from evals.native_eval_session import NativeEvalSession
    from evals.context_spanning_probe import main as probe
    plan = json.loads((args.source/PLAN).read_text())
    if len(plan['cases']) != 10 or plan['word_or_audio_forcing'] is not False:
        raise ValueError('Complete unchanged native development plan required')
    out = args.out / args.role
    out.mkdir(exist_ok=False)
    state = dict(schema='dl-native-paired-development-actor-v1', pid=os.getpid(),
                 role=args.role, proof=PROOFS[args.role], assigned_main_gpu=args.device,
                 assigned_asr_gpu=1-args.device, source_commit=args.source_commit,
                 started_utc=utc(), phase='parity', cases=[], optimizer_updates=0,
                 complete10=False, behavior_approved=False, training_approved=False,
                 provider_credential_present=False)
    atomic(out/'execution.json', state)
    session = NativeEvalSession()
    def run(case, folder):
        if time.time() >= args.deadline:
            raise TimeoutError('Native paired-development deadline reached')
        if (subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip() != args.source_commit
                or subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'],text=True).strip()):
            raise RuntimeError('Pinned native evaluation source changed')
        os.environ['CUDA_VISIBLE_DEVICES'] = (str(args.device) if case['backend']=='observe-only'
                                             else str(args.device)+','+str(1-args.device))
        probe(command(case, PROOFS[args.role], folder, args.asr_python), session=session)
    try:
        run(plan['cases'][0], out/'parity-cold')
        run(plan['cases'][1], out/'parity-intervening')
        run(plan['cases'][0], out/'parity-reused')
        state['parity'] = compare_native(out/'parity-cold', out/'parity-reused')
        atomic(out/'parity.json', state['parity'])
        state['phase'] = 'ordinary'
        atomic(out/'execution.json', state)
        for index, case in enumerate(plan['cases'][:6]):
            run(case, out/('case-%02d'%index))
            state['cases'].append(dict(index=index, completed_utc=utc(), exit_code=0))
            atomic(out/'execution.json', state)
        (out/'ordinary-complete').write_text(utc()+'\n')
        other = args.out / ('candidate7590052' if args.role=='baseline520' else 'baseline520')
        while not (other/'ordinary-complete').exists():
            if (other/'failed').exists():
                raise RuntimeError('Other paired actor failed before full ordinary barrier')
            if time.time() >= args.deadline:
                raise TimeoutError('Full ordinary barrier deadline reached')
            time.sleep(1)
        # Both resident models are idle before isolated actual-ASR work. Only
        # one actor may speak/use ASR at once; the other keeps resettable weights.
        import fcntl
        with (args.out/'asr.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state['phase'] = 'actual_native_asr'
            atomic(out/'execution.json', state)
            for index in range(6,10):
                run(plan['cases'][index], out/('case-%02d'%index))
                state['cases'].append(dict(index=index, completed_utc=utc(), exit_code=0))
                atomic(out/'execution.json', state)
        state.update(phase='complete', complete10=True, finished_utc=utc())
        atomic(out/'execution.json', state)
    except BaseException as error:
        state.update(phase='failed', error_type=type(error).__name__, error=str(error), at_utc=utc())
        atomic(out/'execution.json', state)
        (out/'failed').write_text(type(error).__name__+'\n')
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out', 'asr-python'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--deadline-utc', required=True)
    ap.add_argument('--pod-id', required=True)
    ap.add_argument('--role', choices=tuple(PROOFS))
    ap.add_argument('--device', type=int, choices=(0,1))
    args = ap.parse_args()
    args.deadline = datetime.fromisoformat(args.deadline_utc.replace('Z','+00:00')).timestamp()
    if args.role:
        if args.device is None:ap.error('Actor requires its explicit physical GPU')
        actor(args)
        return
    if args.out.exists() or args.deadline <= time.time()+60:
        ap.error('Fresh output directory and future bounded deadline required')
    args.out.mkdir(parents=True)
    gpus = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],text=True)
    if len(gpus.strip().splitlines()) != 2:
        raise RuntimeError('Exactly two real assigned GPUs required')
    env = os.environ.copy()
    env.pop('OPENAI_API_KEY',None)
    if not env.get('DUPLEX_OPENAI_LOOPBACK_PROXY') or not env.get('DUPLEX_OPENAI_PROXY_RUN'):
        raise RuntimeError('Scoped Mac-only real Responses transport required')
    state = dict(schema='dl-runpod-paired-native10-v1', pid=os.getpid(), pod_id=args.pod_id,
                 source_commit=args.source_commit, plan_sha256=hashlib.sha256((args.source/PLAN).read_bytes()).hexdigest(),
                 started_utc=utc(), deadline_utc=args.deadline_utc, gpus=gpus.strip().splitlines(),
                 phase='independent_restore', children=[], complete_paired20=False,
                 optimizer_updates=0, behavior_approved=False, training_approved=False,
                 provider_credential_transferred=False)
    atomic(args.out/'execution.json',state)
    children = []
    def stop(*_):
        for child, _ in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda a,b: (stop(),sys.exit(128+a)))
    try:
        for device, role in enumerate(PROOFS):
            restore_env = env.copy()
            restore_env['CUDA_VISIBLE_DEVICES'] = str(device)
            cmd = [sys.executable,'-u','-m','portable.context.restore_adapter_state',
                   '--source',str(args.source),'--proof',str(ROOT/'runs'/('context-training-proof-'+PROOFS[role])),
                   '--out',str(args.out/(role+'-independent-restore.json'))]
            with (args.out/(role+'-restore.log')).open('x') as log:
                child = subprocess.Popen(cmd, cwd=args.source, env=restore_env, stdout=log,
                                         stderr=subprocess.STDOUT,start_new_session=True)
                children.append((child,role+'-restore'))
                child.wait(timeout=max(1,args.deadline-time.time()))
                if child.returncode:raise RuntimeError(role+' independent restore failed')
        state['phase'] = 'paired_native_development'
        for device, role in enumerate(PROOFS):
            cmd = [sys.executable,'-u','-m','portable.context.native_development',
                   '--source',str(args.source),'--source-commit',args.source_commit,
                   '--out',str(args.out),'--asr-python',str(args.asr_python),
                   '--deadline-utc',args.deadline_utc,'--pod-id',args.pod_id,
                   '--role',role,'--device',str(device)]
            log = (args.out/(role+'.log')).open('x')
            child = subprocess.Popen(cmd,cwd=args.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            log.close()
            children.append((child,role))
            state['children'].append(dict(pid=child.pid,role=role))
        atomic(args.out/'execution.json',state)
        for child, role in children:
            child.wait(timeout=max(1,args.deadline-time.time()))
            if child.returncode:raise RuntimeError(role+' exited '+str(child.returncode))
        for role in PROOFS:
            result=json.loads((args.out/role/'execution.json').read_text())
            if not result['complete10'] or [c['index'] for c in result['cases']] != list(range(10)):
                raise RuntimeError('Complete paired20 native results missing')
        state.update(phase='complete',complete_paired20=True,finished_utc=utc(),
                     independent_audio_and_behavior_review_required=True)
    except BaseException as error:
        stop()
        state.update(phase='failed',error_type=type(error).__name__,error=str(error),at_utc=utc())
        atomic(args.out/'execution.json',state)
        raise
    atomic(args.out/'execution.json',state)


if __name__ == '__main__':main()
