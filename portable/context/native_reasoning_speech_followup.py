"""Wait for one exact CPU audit, then finalize ALL32 original-waveform verdicts.

Never launches recognition, provider requests, native GPU work or training.
Both original failed qualification and actual recognizer exit evidence remain.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from portable.context.native_preflight_takeover import process
from portable.context.native_control_preflight import git


def ready(actual, birth, exit_value, stage):
    if actual is not None and actual['birth'] != birth:
        raise ValueError('Recognition supervisor identity changed')
    if stage.get('phase') == 'failed_preserved_no_retry':
        raise ValueError('Preserved source staging/verification failed; never retry implicitly')
    if exit_value is None:
        if actual is None or actual['state'] == 'Z':
            raise ValueError('Recognition supervisor terminal without actual child exit receipt')
        return False
    if exit_value != {'actual_exit': 0}:
        raise ValueError('Actual complete recognizer exit0 required')
    if actual is not None and actual['state'] != 'Z': return False
    if actual is not None and os.waitstatus_to_exitcode(actual['exit_status']) != 0:
        raise ValueError('Recognition supervisor actual wait status differs')
    return (stage.get('phase') == 'complete_verified_failed32_on_local_disk'
            and stage.get('passed') is True and stage.get('verify_actual_exit') == 0
            and stage.get('verification') == dict(complete_cases=32, passed=False,
                                                 failed_audit_recomputed_on_remote_cpu=True))


def run(a):
    deadline=datetime.fromisoformat(a.deadline_utc).timestamp()
    if not time.time() < deadline < time.time()+4*3600:
        raise ValueError('Finite future speech-only deadline required')
    def clean():
        if git(a.source,'rev-parse','HEAD')!=a.source_commit or git(a.source,'status','--porcelain'):
            raise ValueError('Pinned clean committed verifier source required')
    clean();a.out.mkdir(parents=True,exist_ok=False)
    state=dict(schema='dl-native-reasoning32-speech-only-followup-v1',pid=os.getpid(),
               source_commit=a.source_commit,upstream_pid=a.upstream_pid,upstream_birth=a.upstream_birth,
               phase='waiting_for_exact_cpu_audit_and_staged_lineage',training_approved=False,
               provider_requests=0,recognition_restarted=False,optimizer_updates=0)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat()
        p=a.out/'status.partial';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(a.out/'status.json')
    save()
    try:
        while time.time()<deadline:
            exit_path=a.upstream_control/'exit.json';stage_path=a.staging_control/'status.json'
            try:
                exit_value=json.loads(exit_path.read_text())if exit_path.exists()else None
                stage=json.loads(stage_path.read_text())if stage_path.exists()else {}
            except json.JSONDecodeError:
                # The producer's small progress/exit files are not atomic.
                # A transient observation is never evidence to restart it.
                time.sleep(1);continue
            if ready(process(a.upstream_pid),a.upstream_birth,exit_value,stage):break
            time.sleep(15)
        else:raise TimeoutError('Bounded speech followup deadline reached; upstream never restarted')
        clean();root=Path(stage['staged_root']);shared=root/'shared'
        command=[str(a.python),'-m','evals.native_reasoning_teacher_speech_v2','finalize',
                 '--source',str(root/'source'),'--quality',str(root/'quality/source-quality.json'),
                 '--batch',str(root/'batch'),'--engine',str(shared/'engine-inputs.json'),
                 '--tokenizer',str(shared/'tokenizer_spm_32k_3.model'),
                 '--failed',str(root/'failed/teacher-speech-quality.json'),
                 '--plan',str(shared/'native-reasoning32-teacher-speech-region-plan-v2.json'),
                 '--worker',str(a.recognition)]
        state.update(phase='recomputing_all32_original_waveform_verdicts',command=command);save()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
                 PYTHONPATH=str(a.packages)+':'+str(a.source),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with(a.out/'finalize.log').open('x')as log:
            result=subprocess.run(command,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,
                                  timeout=max(1,min(300,deadline-time.time())))
        state['finalize_actual_exit']=result.returncode
        target=a.recognition/'teacher-speech-quality.json'
        if not target.exists():raise ValueError('Whole32 derived verdict absent; inspect actual finalization log')
        value=json.loads(target.read_text())
        if (value['complete_cases']!=32 or len(value['results'])!=32
                or value['training_approved']is not False or value['semantic_approved']is not False
                or result.returncode!=(0 if value['passed']else 1)):
            raise ValueError('Actual finalizer exit and complete qualification verdict differ')
        state.update(phase='complete_speech_only_still_unapproved',whole32_speech_passed=value['passed'],
                     passed_cases=sum(r['passed']for r in value['results']),complete_cases=32)
        save();return state
    except BaseException as error:
        state.update(phase='failed_preserved_no_retry',error_type=type(error).__name__,reason=str(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','python','packages','out','upstream-control','staging-control','recognition'):
        p.add_argument('--'+n,type=Path,required=True)
    for n in ('upstream-pid','upstream-birth'):p.add_argument('--'+n,type=int,required=True)
    for n in ('source-commit','deadline-utc'):p.add_argument('--'+n,required=True)
    v=run(p.parse_args());print(json.dumps(v))
