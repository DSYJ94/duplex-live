"""Finite zero-update diagnostic chain, durable and host-Python3.6 compatible."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from evals.native_audio49_chain import BASE,SlotBusy,call,save,now,digest

PARENT_SHA='0310c2563988f8f59d942dce98119230016136f13fc9729ad1c2f398318e7cb5'
STAGES=('predecessor','native16','speech16')
TERMINAL=('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED','SPECIAL_EXIT')


def validate_predecessor(report):
    if (report['schema']!='dl-native-bf16-stream-batch49-diagnostic-v1'
            or not report['all49_complete'] or not report['all_parameters_unchanged']
            or not report['execution_policy_unchanged'] or report['optimizer_updates']!=0
            or report['training_approved'] or report['training_extension_or_adoption_approved']
            or not report['original_reference_approvals_unchanged'] or report['original_reference_threshold']!=.001
            or report['native520_checkpoint_sha256']!=PARENT_SHA
            or len(report['cases'])!=49 or len({x['teacher']['id'] for x in report['cases']})!=49):
        raise ValueError('Complete unchanged49 zero-update predecessor required')


def validate(receipt):
    tree=Path(os.environ['DL_TASK_SOURCE']);source=os.environ['DL_TASK_COMMIT']
    plan=tree/'evals/fixtures/native-precision16-no-tf32-plan-20261020.json'
    predecessor=Path(receipt['predecessor_receipt_path'])
    if predecessor.parent!=BASE/'runs' or not predecessor.name.startswith('native-bf16-stream49-'):
        raise ValueError('Scoped BF16 predecessor receipt required')
    prior=json.loads(predecessor.read_text())
    if receipt.get('cpu_api_preflight_required'):
        item=receipt['cpu_api_preflight'];p=Path(item['path']);pre=json.loads(p.read_text())
        if (digest(p)!=item['sha256'] or not pre['passed'] or pre['source']!=receipt['source']
                or pre['allocation_job']!=receipt['scheduler_watcher_job']
                or pre['tests_run']!=7 or pre['cuda_initialized'] or not pre['cpu_backend_contract_only']):
            raise ValueError('Exact successful CPU backend compatibility preflight required')
    if (receipt['source']!=source or receipt['executor']!='LOCAL'
            or receipt['optimizer_updates']!=0 or receipt['training_extension_or_adoption_approved']
            or digest(predecessor)!=receipt['predecessor_receipt_sha256']
            or prior['source']!=receipt['predecessor_source'] or prior['optimizer_updates']!=0
            or not receipt['jobs']['predecessor'].isdigit()
            or prior['jobs']['bf16_stream49']!=receipt['jobs']['predecessor'] or receipt['lm_matmul_tf32'] is not False
            or receipt['scheduler_watcher_job']!=os.environ['DL_CHAIN_WATCHER_JOB']
            or call(['git','-C',str(tree),'rev-parse','HEAD']).strip()!=source
            or call(['git','-C',str(tree),'status','--porcelain']).strip()
            or digest(plan)!=receipt['plan_sha256']):
        raise ValueError('Immutable LOCAL zero-update diagnostic identity required')


def evidence(receipt,stage):
    if stage=='predecessor':
        path=BASE/'runs'/('native-audio49-training-preflight-'+receipt['jobs']['predecessor'])/'bf16-stream49-diagnostic.json'
        validate_predecessor(json.loads(path.read_text()))
        receipt['bf16_stream49_sha256']=digest(path)
    elif stage=='native16':
        # Speech audits are also required after failed native execution; retain failures.
        receipt['speech_must_retain_all16_cases']=True
    else:raise ValueError('No further transition after speech')


def submit(path,receipt,stage):
    validate(receipt)
    if stage not in ('predecessor','native16'):raise ValueError('No submission after terminal speech')
    following='native16' if stage=='predecessor' else 'speech16'
    if following in receipt['jobs'] or following in receipt.get('submission_attempts',{}):
        raise ValueError('Transition already attempted; never resubmit an uncertain sbatch')
    active=call(['squeue','--all','-u',os.environ['USER'],'-h','-o','%i']).split()
    watcher=os.environ['DL_CHAIN_WATCHER_JOB']
    if any(j!=watcher for j in active) or len(active)>1:
        raise SlotBusy('Active scheduler IDs '+repr(active)+'; watcher '+watcher+'; no submission attempted')
    evidence(receipt,stage)
    receipt.setdefault('submission_attempts',{})[following]=dict(status='submitting',at_utc=now())
    save(path,receipt)
    env={k:v for k,v in os.environ.items() if not k.startswith(('SBATCH_','SLURM_'))}
    if following=='speech16':env['NATIVE_PRECISION_NO_TF32_SESSION_JOB']=receipt['jobs']['native16']
    script='native-precision-no-tf32-session.slurm' if following=='native16' else 'native-precision-no-tf32-asr.slurm'
    answer=call(['sbatch','--hold','--parsable','--export=ALL','--no-requeue',str(Path(env['DL_TASK_SOURCE'])/'evals'/script)],env).strip()
    job=answer.split(';')[0]
    if not job.isdigit():raise ValueError('Uncertain sbatch response; inspect authoritative scheduler, do not retry')
    receipt['jobs'][following]=job;receipt['submission_attempts'][following].update(status='held',job=job);save(path,receipt)
    settings=call(['scontrol','show','job',job]);receipt.setdefault('held_settings',{})[following]=settings;save(path,receipt)
    if 'TimeLimit=00:30:00' not in settings:raise ValueError('Unexpected held allocation')
    if following=='native16':
        if 'Reservation=interactive' not in settings or 'TresPerNode=gres/gpu:2' not in settings:raise ValueError('Unexpected native GPU allocation')
    elif 'gres/gpu' in settings:raise ValueError('Independent speech must use CPU only')
    call(['scontrol','release',job]);receipt.setdefault('released',[]).append(job)
    receipt['submission_attempts'][following]['status']='released';save(path,receipt)
    print('RELEASED',following,job,flush=True)


def watch(path):
    with Path(str(path)+'.watch.lock').open('a') as watcher_lock:
        fcntl.flock(watcher_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for _ in range(120):
            with Path(str(path)+'.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                receipt=json.loads(path.read_text());validate(receipt)
                stage=next(s for s in reversed(STAGES) if s in receipt['jobs']);job=receipt['jobs'][stage]
                try:answer=call(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode'])
                except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
                    print('OBSERVATION_FAILED_NO_SUBMISSION',job,str(error),flush=True);answer=''
                rows=[x.split('|') for x in answer.splitlines() if x.startswith(job+'|')]
                receipt['watcher_heartbeat_utc']=now()
                if len(rows)==1:
                    state,code=rows[0][1:3]
                    receipt.setdefault('slurm_observed',{})[stage]=dict(job=job,state=state,exit_code=code,at_utc=now());save(path,receipt)
                    print(stage,job,state,code,flush=True)
                    if state.split(' ',1)[0] in TERMINAL:
                        if stage=='speech16' or (stage=='predecessor' and (state,code)!=('COMPLETED','0:0')):
                            receipt['watcher_terminal']=dict(stage=stage,job=job,state=state,exit_code=code,at_utc=now());save(path,receipt)
                            if (state,code)!=('COMPLETED','0:0'):raise RuntimeError('Diagnostic stage failed; retained without restart')
                            return
                        try:submit(path,receipt,stage)
                        except SlotBusy as error:print('WAIT_ACTIVE_SCHEDULER_SLOT',str(error),flush=True)
                else:
                    save(path,receipt);print('WAIT_ACCOUNTING',job,flush=True)
            time.sleep(60)
        raise RuntimeError('Finite watcher reached its observation budget; inspect before recovery')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--receipt',type=Path,required=True)
    watch(ap.parse_args().receipt)
