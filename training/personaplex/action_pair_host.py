"""Bounded sequential paired phase; no scheduler submission or implicit retry."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from .expanded_guard import AUTH, PAIR_AUTH, ACCOUNT, KEY, sha, parse_job, stamp, validate_job
from .expanded_evaluation_host import parent_provider
from .native_feature_host import feature_command, archive


def execute_pairs(run, preserve):
    """A failed fit/backup/restore stops all later work; never repeat an arm."""
    receipts = {}
    for mode in ('readout','bias'):
        run(mode, 'fit')
        receipts[mode+'/fit'] = preserve(mode, 'fit', 128)
        run(mode, 'evaluate')
        receipts[mode+'/evaluate'] = preserve(mode, 'evaluate', 0)
    return receipts


def pair_command(work,environment,backup,inputs):
    command = feature_command(work,True,dict(environment,HF_HOME='/projects/u6ada/duplex-live/hf-cache'),backup)
    protected = [Path('/home/u6ada/dsyj94.u6ada/duplex-native-feature-backups-v1'),
                 Path('/projects/u6ada/duplex-live/native-feature-restore-20261117-v1/cache'),
                 Path('/projects/u6ada/duplex-live/native-restore-only-20261117-v1'),
                 Path('/projects/u6ada/duplex-live/native-expanded1072-phase-v1'),inputs]
    mounts = []
    for path in protected:
        for logical in sorted({str(path),str(path.resolve())}):
            mounts.extend(['--bind',str(path.resolve())+':'+logical+':ro'])
    position = command.index('--pwd')
    command[position:position] = mounts
    return command


def admission(a, plan_path):
    actual = parse_job(subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']], universal_newlines=True, timeout=30))
    raw = subprocess.check_output(['sacct','-A',ACCOUNT,'-S','2026-10-30','-X','-n','-P','-o','JobIDRaw,JobName%100,State,ElapsedRaw,AllocTRES,ReqTRES,TimelimitRaw'], universal_newlines=True, timeout=30)
    prior = reserved = 0
    for line in raw.splitlines():
        f = line.split('|')
        if len(f) < 7 or f[0] == actual['JobId']:
            continue
        ident,name,state,elapsed,allocated,requested,limit = f[:7]
        tres = dict(v.split('=',1) for v in (allocated or requested).split(',') if '=' in v)
        count, used = int(tres.get('gres/gpu',0)), int(elapsed)
        prior += count*used
        if state in ('RUNNING','PENDING','CONFIGURING'):
            if name.startswith('duplex-') and count:
                raise ValueError('Never overlap another valid voice allocation')
            if not limit.isdigit():
                raise ValueError('Actual bounded concurrent reservation required')
            reserved += count*max(0,int(limit)*60-used)
    quota = subprocess.check_output(['quota','-w','-v'], universal_newlines=True, timeout=20)
    q = next(r.split() for r in quota.splitlines() if r.strip().startswith('data1.vastp2.isambard.ac.uk:/p2/home '))
    used, hard = int(q[1])*1024, int(q[3])*1024
    if hard-used < 10*1024**3:
        raise ValueError('Full paired independent backup needs10GiB actual headroom')
    lease = dict(paired_authorization_sha256=sha(PAIR_AUTH), provider='Slurm', resource_key=KEY,
                 account=ACCOUNT, authorization_sha256=sha(AUTH), GPU_count=1,
                 job_id=actual['JobId'], node=actual['NodeList'], source=a.source,
                 start_unix=stamp(actual['StartTime']), stop_unix=stamp(actual['EndTime']),
                 prior_award_GPU_seconds=prior, other_reserved_GPU_seconds=reserved,
                 home_used_bytes=used, home_hard_bytes=hard, action_plan_sha256=sha(plan_path))
    validate_job(actual, lease, time.time())
    return lease, raw+'\n'+quota


def main(a):
    scope = json.loads(PAIR_AUTH.read_text())
    if scope['status'] != 'OWNER_AUTHORIZED_PAIRED_RET':
        raise ValueError('Prepared phase cannot launch without its new scoped authorization')
    if subprocess.check_output(['git','rev-parse','HEAD'], universal_newlines=True).strip() != a.source:
        raise ValueError('Exact immutable committed phase source required')
    parent = parent_provider('8537504')
    if parent.get('JobState') != 'COMPLETED' or parent.get('ExitCode') != '0:0':
        raise ValueError('Completed restore parent required; never overlap it')
    a.backup.mkdir(parents=True,exist_ok=True)
    a.root.mkdir(exist_ok=False)
    lock = (a.root/'.controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = json.loads(a.plan.read_text())
    prep = json.loads((a.inputs/'input-preparation.json').read_text())
    for filename, key in (('control-cases.json','control32_sha256'),('dev-cases.json','dev64_sha256')):
        if sha(a.inputs/filename) != prep[key]:
            raise ValueError('Original complete input cohort changed')
        plan[key] = prep[key]
    plan['source'] = a.source
    path = a.root/'plan.json'
    path.write_text(json.dumps(plan,indent=2)+'\n')
    lease, raw = admission(a,path)
    folder = a.root/('job-'+lease['job_id'])
    folder.mkdir()
    lease_path = folder/'lease.json'
    lease_path.write_text(json.dumps(lease,indent=2)+'\n')
    (folder/'provider-admission.txt').write_text(raw)
    env = dict(os.environ,DUPLEX_LIVE_GUARD_FILE=str(folder/'guard.json'),DUPLEX_LIVE_LEASE_FILE=str(lease_path))
    watch = subprocess.Popen(['python3','-u','-m','training.personaplex.expanded_guard','--lease',str(lease_path),'--output',str(folder/'guard.json')],env=env)
    try:
        until = time.time()+180
        while not (folder/'guard.json').exists():
            if watch.poll() is not None or time.time() > until:
                raise ValueError('Actual independent allocation guard unavailable')
            time.sleep(2)
        inside = {k:env[k] for k in ('DUPLEX_LIVE_GUARD_FILE','DUPLEX_LIVE_LEASE_FILE','SLURM_JOB_ID','SLURM_JOB_ACCOUNT')}
        command = pair_command(Path.cwd(),inside,a.backup,a.inputs)
        def run(mode,stage):
            subprocess.run(command+['training.personaplex.expanded_gpu','training.personaplex.action_pair_worker',stage,'--mode',mode,'--root',str(a.root),'--lease',str(lease_path),'--plan',str(path),'--inputs',str(a.inputs),'--source',a.source],env=env,check=True)
            value = json.loads((a.root/mode/stage/'terminal.json').read_text())
            if stage == 'fit' and (value['actual_new_readout_updates'] != 128 or value['backbone_updates'] != 0):
                raise ValueError('Exactly128 new head updates per arm required')
            if stage == 'evaluate' and (not value['passed'] or value['outputs'] != 96 or value['optimizer_updates'] != 0):
                raise ValueError('Fresh native restore and complete96 outputs required')
        def preserve(mode,stage,updates):
            receipt = archive(a.root/mode/stage,a.backup/('job-'+lease['job_id'])/mode/stage,optimizer_updates=updates)
            (folder/(mode+'-'+stage+'-backup.json')).write_text(json.dumps(receipt,indent=2)+'\n')
            return receipt
        receipts = execute_pairs(run,preserve)
        (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,source=a.source,backups=receipts,new_readout_updates=256,native_backbone_updates=0,complete_native_outputs=192,baseline_outputs_regenerated=False,product_accepted=False),indent=2)+'\n')
    finally:
        watch.terminate()
        watch.wait(timeout=30)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root','backup','inputs','plan'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--source',required=True)
    main(p.parse_args())
