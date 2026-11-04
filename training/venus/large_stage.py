"""One bounded CPU staging interval, then STOP; never allocates a GPU."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from . import controller as c
from .large_guard import AUTH, ROOT, authority
from .large_inputs import PARENT_SHA, sha

R = ROOT/'stage'
ZONE = 'us-central1-a'
NAME = 'duplex-venus-large-stage-20261104-v65'
DISK = 'duplex-venus-large-data-20261104-a-v65'
REMOTE = '/workspace/migration/venus-large-capacity-v65'
LOCAL = Path('/Users/daryl/Documents/duplex-native-recovery-20261102/context-v52')
PARENT = LOCAL/'candidate/step-0060.pt'


def save(name, value):
    (R/name).write_text(json.dumps(value, indent=2)+'\n')


def stop_exact(ident):
    vm = c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
    if vm['id'] != ident: raise ValueError('CPU identity changed; never stop a different machine')
    if vm['status'] != 'TERMINATED':
        c.run(['gcloud','compute','instances','stop',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet'],150)


def create_disk(snap):
    return json.loads(c.run(['gcloud','compute','disks','create',DISK,'--zone='+ZONE,
              '--source-snapshot='+snap['name'],'--size=250GB','--type=pd-balanced',
              '--labels=duplex-scope=large-v65','--project='+c.PROJECT,'--quiet','--format=json'],900))


def guard():
    authority(); lease=json.loads((R/'lease.json').read_text())
    try:
        while True:
            vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
            if vm['status']=='TERMINATED':return
            if (vm['id']!=lease['vm_id'] or not vm['machineType'].endswith('/e2-standard-4')
                    or vm.get('guestAccelerators') or time.time()>=lease['stop_unix']
                    or c.run(['/bin/ps','-p',str(lease['controller_pid']),'-o','lstart=']).strip()!=lease['controller_birth']):
                raise ValueError('CPU stage identity, controller or bounded clock failed')
            if any(x['id']!=vm['id'] and x['status'] not in ('TERMINATED','STOPPED') for x in c.cloud('compute','instances','list')):
                raise ValueError('CPU staging must not overlap an active GPU')
            save('guard.json',dict(vm_id=vm['id'],checked_unix=time.time(),expires_unix=min(time.time()+60,lease['stop_unix'])))
            time.sleep(15)
    finally:
        stop_exact(lease['vm_id'])


def verify(a):
    """CPU-only validation of the exact staged allowlist, never model inference."""
    import torch
    from .large_inputs import Corpus
    root=Path(a.remote_root)
    corpus=Corpus(root/'cache',action_index=root/'cache/action-index.json',
                  admission='docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-V3-2026-11-02.json')
    if sha(root/'parent466.pt')!=PARENT_SHA:raise ValueError('Preserved parent466 hash differs')
    parent=torch.load(root/'parent466.pt',map_location='cpu',weights_only=False)
    if parent['step']!=60 or parent['metadata']['parent_training_steps']!=406 or parent['optimizer']['update']!=60:
        raise ValueError('Actual full optimizer state466 required')
    counts={f:0 for f in ('TRAIN','DEV')};max_tokens=0
    for row in corpus.rows:
        example=corpus.read(row);counts[row['fold']]+=1;max_tokens=max(max_tokens,len(example['embeddings']))
    model=Path(c.MODEL);config=json.loads((model/'config.json').read_text())
    if (config['hidden_size']!=4096 or config['intermediate_size']!=12288
            or config['num_hidden_layers']!=36):raise ValueError('Exact pinned model geometry required')
    # No uploaded credentials, private conversations or quarantined core files
    # are allowed in the staged payload. Original public/runtime files on the
    # approved source image stay in place; provider credentials remain on Mac.
    forbidden={'.private','credentials','owner-conversations','session-token','API keys.md','core'}
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if forbidden & set(relative.parts) or path.name.startswith('core.'):
            raise ValueError('Excluded file present in stage: '+str(relative))
    result=dict(passed=True,source=a.source,all304_native_hashes_targets_and_full_inputs_verified=True,
                folds=counts,largest_native_input_tokens=max_tokens,parent_sha256=PARENT_SHA,
                original_parent_optimizer_updates=466,new_optimizer_updates=0,
                GPU_model_loaded=False,python=sys.version.split()[0],torch=torch.__version__,
                model_geometry_verified=True,excluded_payload_names_absent=True)
    (root/'stage-verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def main(resume_precreate=False):
    from .source_controller import prepare_ssh
    from .tts_placement import snapshot, observe
    from portable.venus.guard import bounded_runpod_read
    authority()
    if R.exists() and not resume_precreate:raise ValueError('Observe existing stage process; never restart its claim')
    if resume_precreate and (not (R/'claim.json').exists() or (R/'provider-start.json').exists() or (R/'disk.json').exists()):
        raise ValueError('Only a confirmed failure before disk/CPU creation may resume')
    price=json.loads(Path('runs/product-progress-watch/current-cpu-stage-price-20261104.json').read_text())
    if (time.time()-datetime.fromisoformat(price['at_utc']).timestamp()>3600
            or not 0<price['hourly_usd']<=.5 or price['hourly_usd']*4>2):
        raise ValueError('Fresh full CPU+250GB price inside4h/2USD required')
    if sha(PARENT)!=PARENT_SHA:raise ValueError('Exact locally preserved parent required')
    source=c.run(['git','rev-parse','HEAD']).strip()
    if c.run(['git','-c','core.filemode=false','status','--porcelain','--','training/venus','portable/venus','data/venus']).strip():
        raise ValueError('Git-deployed source must be committed before any resource mutation')
    current=c.cloud('compute','instances','list')
    if any(x['status'] not in ('TERMINATED','STOPPED') for x in current):raise ValueError('All previous compute must be stopped')
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip())
    if isinstance(pods,dict):pods=pods['pods']
    if any(p['status'] not in ('EXITED','STOPPED','TERMINATED') for p in pods):raise ValueError('Runpod GPU must be stopped')
    if any(x['name'] in (NAME,DISK) for x in current+c.cloud('compute','disks','list')):
        raise ValueError('Existing named resource must be observed, never duplicated')
    if resume_precreate:
        # The named resources were freshly absent above. This is not a second
        # CPU interval or a new storage clock; preserve the original claim.
        prior=json.loads((R/'claim.json').read_text())
        failure=json.loads((R/'precreate-failure.json').read_text())
        if failure['reason']!='gcloud disks create rejects --async before provider request' or failure['CPU_intervals']!=0:
            raise ValueError('Exact terminal pre-provider failure proof required')
        c.events=json.loads((R/'events.json').read_text())
        with (R/'precreate-recovery-claim.json').open('x') as f:
            json.dump(dict(previous_claim_source=prior['source'],source=source,pid=os.getpid(),
                           original_claim_preserved=True,named_resources_freshly_absent=True),f)
        deadline=json.loads((R/'storage-deadline.json').read_text())['storage_deadline_unix']
        bundle=R/'source-precreate-recovery.bundle'
    else:
        R.mkdir(parents=True);c.events=[]
        with (R/'claim.json').open('x') as f:json.dump(dict(source=source,pid=os.getpid(),authority_sha256=sha(AUTH)),f)
        deadline=time.time()+86400;save('storage-deadline.json',dict(storage_started_unix=deadline-86400,storage_deadline_unix=deadline))
        bundle=R/'source.bundle'
    c.R=R;c.NAME=NAME
    if time.time()+14400>=deadline:raise ValueError('Original immutable storage window insufficient')
    c.run(['git','bundle','create',str(bundle),'HEAD'],120)
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())]);watch=None;vm=None
    try:
        if resume_precreate:
            previous=json.loads((R/'snapshot-ready.json').read_text())
            snap=c.cloud('compute','snapshots','describe',previous['name'])
            if snap['id']!=previous['id'] or snap['status']!='READY' or snap['sourceDiskId']!='4375908109499109167':
                raise ValueError('Same owned already recovered snapshot required')
        else:snap=snapshot()
        c.record('one_CPU_stage_clone_request',source=source,disk_name=DISK,maximum_GiB=250,storage_deadline_unix=deadline)
        # This gcloud command has no --async flag. Wait on this one request;
        # timeout means inspect the SAME named disk, never submit again.
        result=create_disk(snap)
        save('disk-create-terminal.json',result)
        disk=c.cloud('compute','disks','describe',DISK,'--zone='+ZONE);save('disk.json',disk)
        if disk['sourceSnapshotId']!=snap['id'] or int(disk['sizeGb'])!=250 or disk.get('users'):
            raise ValueError('Exact approved source clone required')
        boot=R/'bootstrap.sh'
        boot.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-large65-cpu-stop --on-active=14300s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=4h')
        op=c.cloud('compute','instances','create',NAME,'--zone='+ZONE,'--machine-type=e2-standard-4',
                   '--provisioning-model=STANDARD','--no-restart-on-failure','--no-service-account','--no-scopes',
                   '--tags=duplex-venus-train-bounded','--disk=name='+DISK+',boot=yes,auto-delete=no,mode=rw',
                   '--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE',
                   '--metadata-from-file=startup-script='+str(boot),'--max-run-duration=14400s',
                   '--instance-termination-action=STOP','--async')
        op=op[0] if isinstance(op,list) else op;save('create.json',op);op=observe(op,ZONE)
        if op.get('error'):raise RuntimeError(str(op['error']))
        vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);save('provider-start.json',vm)
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+14400
        save('lease.json',dict(vm_id=vm['id'],disk_id=disk['id'],stop_unix=end,controller_pid=os.getpid(),
             controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip()))
        watch=subprocess.Popen([sys.executable,'-u','-m','training.venus.large_stage','--guard'],
                               stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT)
        prepare_ssh(vm,ZONE,'65stage',require_gpu=False)
        until=time.time()+60
        while not (R/'guard.json').exists():
            if watch.poll() is not None or time.time()>until:raise RuntimeError('Independent CPU guard did not arm')
            time.sleep(1)
        if c.props('duplex-large65-cpu-stop.timer')['ActiveState']!='active':raise RuntimeError('CPU guest STOP timer absent')
        c.remote=c.ROOT+'/duplex-large-'+source[:12]
        c.sync(bundle,c.ROOT+'/large65-source.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/large65-source.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+source,120)
        c.ssh('mkdir -p '+REMOTE)
        c.sync(str(LOCAL/'cache')+'/',REMOTE+'/cache/',900)
        c.sync(PARENT,REMOTE+'/parent466.pt',180)
        command='/workspace/migration/venus-runtime-v1/venv/bin/python -u -m training.venus.large_stage --verify --remote-root '+REMOTE+' --source '+source
        unit='duplex-venus-large65-stage-verify'
        c.start_unit(unit,command,900)
        until=time.time()+960
        while True:
            state=c.props(unit)
            if state['SubState']=='exited' or state['ActiveState'] in ('failed','inactive'):break
            if watch.poll() is not None or time.time()>until:raise TimeoutError('Observe same CPU verification; never relaunch')
            time.sleep(5)
        save('verification-terminal.json',state)
        journal=c.ssh('sudo journalctl -u '+unit+' --no-pager -o cat');(R/'verification.log').write_text(journal)
        if state['Result']!='success' or state['ExecMainStatus']!='0':raise RuntimeError('Actual CPU stage verification failed; inspect preserved journal')
        raw=c.ssh('cat '+REMOTE+'/stage-verified.json');(R/'stage-verified.json').write_text(raw)
        if json.loads(raw)['source']!=source or not json.loads(raw)['passed']:raise ValueError('Actual source-bound full stage proof required')
        if c.ssh('sha256sum '+REMOTE+'/stage-verified.json').split()[0]!=sha(R/'stage-verified.json'):
            raise ValueError('Stage result copyback hash differs')
        c.record('complete304_CPU_stage_verified',source=source,vm_id=vm['id'],disk_id=disk['id'],GPU_updates=0)
    finally:
        found=[x for x in c.cloud('compute','instances','list') if x['name']==NAME]
        if found:
            stop_exact(found[0]['id'])
            final=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);save('provider-final.json',final)
            c.record('CPU_stage_stopped',vm_id=final['id'],disk_name=DISK,source=source,
                     actual_seconds=(datetime.fromisoformat(final['lastStopTimestamp'])-datetime.fromisoformat(final['lastStartTimestamp'])).total_seconds())
        if watch:
            try:watch.wait(timeout=30)
            except subprocess.TimeoutExpired:watch.terminate()
        awake.terminate()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--guard',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--resume-precreate',action='store_true');p.add_argument('--remote-root');p.add_argument('--source');a=p.parse_args()
    guard() if a.guard else verify(a) if a.verify else main(a.resume_precreate)
