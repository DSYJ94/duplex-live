"""One value-selected A10040 Flex interval after completed CPU staging.

Native preparation, independent restore, complete-epoch main/control and full
paired collection run serially. Every failure stops compute and preserves disk;
this controller never retries an allocation or repeats a saved optimizer step.
"""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from . import controller as c
from .large_guard import AUTH, ROOT, KEY, OLD_USD, authority
from .large_inputs import PARENT_SHA, MANIFEST_SHA, sha
from .large_stage import NAME as CPU, DISK, ZONE, REMOTE

R=ROOT/'gpu-01'
NAME='duplex-venus-large-gpu-20261104-a-v65'
OUTPUT=REMOTE+'/native-v65'
BACKUP=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v65')


def save(name,value):(R/name).write_text(json.dumps(value,indent=2)+'\n')


def observe(op):
    until=time.time()+1020
    while op['status']!='DONE':
        if time.time()>until:raise TimeoutError('Observe SAME queued provider operation; never create again')
        time.sleep(5);op=c.cloud('compute','operations','describe',op['name'],'--zone='+ZONE)
        save('latest-operation.json',op)
    return op


def copyback():
    # Hash only this new native run, never rehash the old81GB archive.
    script="""import hashlib,json,os
from pathlib import Path
r=Path(%r)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
print(json.dumps(dict(files=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()],links=[dict(path=str(p.relative_to(r)),target=os.readlink(p)) for p in sorted(r.rglob('*')) if p.is_symlink()])))
"""%OUTPUT
    inventory=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote(script),600));save('remote-inventory.json',inventory)
    if sum(x['bytes'] for x in inventory['files'])>100_000_000_000:raise ValueError('Approved100GB artifact limit exceeded; preserve disk')
    BACKUP.mkdir(parents=True,exist_ok=False)
    argv=['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private',
          '--exclude=credentials','--exclude=owner-conversations','--exclude=API keys.md','--exclude=core','--exclude=core.*',
          '-e',shlex.join(c.ssh_argv[:-1]),c.ssh_argv[-1]+':'+OUTPUT+'/',str(BACKUP)+'/']
    c.run(argv,1800)
    for row in inventory['files']:
        path=BACKUP/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Native result backup hash differs')
    for row in inventory['links']:
        path=BACKUP/row['path']
        if not path.is_symlink() or os.readlink(path)!=row['target']:raise ValueError('Native result link differs')
    save('copyback-verified.json',dict(all_hashes_verified=True,files=len(inventory['files']),
         bytes=sum(x['bytes'] for x in inventory['files']),links=len(inventory['links']),
         backup=str(BACKUP),inventory_sha256=sha(R/'remote-inventory.json')))


def main():
    from .source_controller import prepare_ssh
    from portable.venus.guard import bounded_runpod_read
    authority()
    if R.exists():raise ValueError('Observe the real existing GPU controller; never duplicate')
    stage=ROOT/'stage'
    proof=json.loads((stage/'stage-verified.json').read_text())
    previous=json.loads((stage/'provider-final.json').read_text())
    if not proof['passed'] or previous['status']!='TERMINATED':raise ValueError('Complete CPU stage and actual STOP required')
    clock=json.loads((stage/'storage-deadline.json').read_text())
    duration=43100
    if time.time()+duration+1800>=clock['storage_deadline_unix']:raise ValueError('Original storage clock insufficient')
    price=json.loads(Path('runs/product-progress-watch/current-capacity-prices-20261104-launch-v65.json').read_text())
    hourly=price['options']['GCP_A10040_FLEX_US_12CPU_85GiB_250PD']
    if not 0<hourly<=8 or time.time()-datetime.fromisoformat(price['at_utc']).timestamp()>3600:
        raise ValueError('Refresh actual full GPU component price before allocation')
    source=c.run(['git','rev-parse','HEAD']).strip()
    if c.run(['git','-c','core.filemode=false','status','--porcelain','--','training/venus','portable/venus']).strip():
        raise ValueError('Source must be Git committed before GPU deployment')
    instances=c.cloud('compute','instances','list')
    if any(v['status'] not in ('TERMINATED','STOPPED') for v in instances):raise ValueError('All compute must actually be stopped')
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip())
    if isinstance(pods,dict):pods=pods['pods']
    if any(v['status'] not in ('EXITED','STOPPED','TERMINATED') for v in pods):raise ValueError('No overlapping Runpod GPU')
    quotas={v['metric']:v for v in c.cloud('compute','regions','describe','us-central1')['quotas']}
    q=quotas['PREEMPTIBLE_NVIDIA_A100_GPUS']
    if q['usage']!=0 or q['limit']<1:raise ValueError('Actual available FlexA100 quota required; quota is not stock')
    rules={v['name']:v for v in c.cloud('compute','firewall-rules','list')}
    allow=rules['duplex-venus-train-iap-v1'];deny=rules['duplex-venus-train-deny-v1']
    if (allow['sourceRanges']!=['35.235.240.0/20'] or allow['allowed']!=[dict(IPProtocol='tcp',ports=['22'])]
            or deny['denied']!=[dict(IPProtocol='all')] or deny['sourceRanges']!=['0.0.0.0/0']
            or 'duplex-venus-train-bounded' not in allow['targetTags'] or 'duplex-venus-train-bounded' not in deny['targetTags']
            or allow['priority']>=deny['priority']):raise ValueError('Existing authenticated IAP/restricted-ingress fence required')
    disk=c.cloud('compute','disks','describe',DISK,'--zone='+ZONE)
    olddisk=json.loads((stage/'disk.json').read_text())
    if disk['id']!=olddisk['id'] or int(disk['sizeGb'])!=250:raise ValueError('SAME staged250GB disk required')
    if any(v['name']==NAME for v in instances):raise ValueError('Existing GPU name must be observed')
    R.mkdir();c.R=R;c.NAME=NAME;c.events=[]
    with (R/'claim.json').open('x') as stream:json.dump(dict(source=source,pid=os.getpid(),authority_sha256=sha(AUTH)),stream)
    c.run(['git','bundle','create',str(R/'source.bundle'),'HEAD'],120)
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())]);watch=None;created=False;end=None
    try:
        current=c.cloud('compute','instances','describe',CPU,'--zone='+ZONE)
        if current['id']!=previous['id'] or current['status']!='TERMINATED':raise ValueError('Exact stopped staging CPU required')
        c.run(['gcloud','compute','instances','delete',CPU,'--zone='+ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
        if c.cloud('compute','disks','describe',DISK,'--zone='+ZONE).get('users'):raise ValueError('Staged disk still attached')
        boot=R/'bootstrap.sh';boot.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-large65-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=14h')
        created=True
        op=c.cloud('compute','instances','create',NAME,'--zone='+ZONE,'--machine-type=a2-highgpu-1g',
             '--provisioning-model=FLEX_START','--request-valid-for-duration=900s','--maintenance-policy=TERMINATE',
             '--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded',
             '--disk=name='+DISK+',boot=yes,auto-delete=no,mode=rw',
             '--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE',
             '--metadata-from-file=startup-script='+str(boot),'--max-run-duration='+str(duration)+'s',
             '--instance-termination-action=STOP','--async')
        op=op[0] if isinstance(op,list) else op;save('create.json',op)
        c.record('one_A10040_Flex_request',source=source,operation=op['name'],maximum_queue_seconds=900)
        op=observe(op);save('allocation-terminal.json',op)
        if op.get('error'):raise RuntimeError(json.dumps(op['error']))
        vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);save('provider-start.json',vm)
        if vm['status']!='RUNNING':raise ValueError('Actual running provider state required')
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+duration
        prepare_ssh(vm,ZONE,'65gpu')
        c.remote=c.ROOT+'/duplex-large-'+source[:12]
        c.sync(R/'source.bundle',c.ROOT+'/large65-gpu-source.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/large65-gpu-source.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+source,120)
        plan=dict(resource_key=KEY,authorization_sha256=sha(AUTH),parent_sha256=PARENT_SHA,cache_manifest_sha256=MANIFEST_SHA,
             maximum_all_branch_updates=984,external_model_API_calls=0,prior_project_USD=OLD_USD,
             vm_id=vm['id'],disk_id=disk['id'],zone=ZONE,creation_timestamp=vm['creationTimestamp'],
             **clock,previous_intervals=[],maximum_running_seconds=duration,disk_size_gib=250,
             machine='a2-highgpu-1g',provisioning='FLEX_START',optimizer_work_started=False,vm_name=NAME,disk_name=DISK)
        plan_path=ROOT/'interval-01.json'
        with plan_path.open('x') as f:json.dump(plan,f,indent=2)
        lease=dict(plan,execution_plan_path=str(plan_path),execution_plan_sha256=sha(plan_path),
             prior_capacity_GPU_seconds_upper=0,verified_hourly_price_usd=hourly,stop_unix=end,
             prior_gpu_seconds=89151,prior_scope_usd=OLD_USD,controller_pid=os.getpid(),
             controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        save('lease.json',lease)
        c.ssh('mkdir -p '+c.remote+'/'+str(ROOT))
        c.sync(plan_path,c.remote+'/'+str(plan_path));lease_path=c.ROOT+'/large65-lease.json';c.sync(R/'lease.json',lease_path)
        watch=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(R/'lease.json'),
             '--ssh-argv',str(R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key',
             '--status',str(R/'guard.json'),'--stop-file',str(R/'stop-requested')],stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT)
        c.guard=watch
        until=time.time()+100
        while not (R/'guard.json').exists():
            if watch.poll() is not None or time.time()>until:raise RuntimeError('Independent GPU guard failed to arm')
            time.sleep(1)
        c.ssh('sudo systemd-run --unit=duplex-large65-guest --property=RuntimeMaxSec='+str(duration)+' --working-directory='+c.remote+
              ' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lease_path)
        if c.props('duplex-large65-guest')['ActiveState']!='active':raise RuntimeError('Guest GPU guard absent')
        c.ssh('sudo systemctl stop duplex-large65-bootstrap-stop.timer')
        if c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').strip():raise ValueError('GPU already has a model process')
        c.record('actual_A10040_running_both_guards_live',vm_id=vm['id'],disk_id=disk['id'],source=source,native_STOP_unix=end)
        env='env DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+lease_path+' HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache '
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python'
        common=' --model '+c.MODEL+' --cache '+REMOTE+'/cache --action-index '+REMOTE+'/cache/action-index.json --data-admission '+c.remote+'/docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-V3-2026-11-02.json --opal-source '+c.ROOT+'/opal-upstream-v34 --lease '+lease_path+' --source '+source
        def execute(unit,module,args,seconds):
            if seconds<1:raise TimeoutError('No authorized running allowance remains')
            c.start_unit(unit,env+py+' -u -m '+module+args,seconds);c.wait_unit(unit,min(time.time()+seconds+15,end-180))
        execute('duplex-large65-prepare','training.venus.large_native',common+' --stage prepare --output '+OUTPUT+' --parent '+REMOTE+'/parent466.pt',1200)
        execute('duplex-large65-restore','training.venus.large_native',common+' --stage restore --output '+OUTPUT,900)
        for arm in ('main','control'):
            execute('duplex-large65-train-'+arm,'training.venus.large_train',common+' --arm '+arm+' --root '+OUTPUT+' --parent '+REMOTE+'/parent466.pt',int(end-time.time()-1800))
            execute('duplex-large65-final-'+arm,'training.venus.large_native',common+' --stage final-restore --output '+OUTPUT+'/'+arm,900)
        c.record('both_complete_training_arms_and_independent_final_restores',source=source,adopted=False)
        evaluate=' --root '+OUTPUT+' --model '+c.MODEL+' --lease '+lease_path+' --source '+source+' --plan '+c.remote+'/evals/fixtures/venus-control-repair-source-v2-20261027.json --timelines '+c.OUT+'/paced-timelines-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --word-quality '+c.OUT+'/paced-word-quality-v22.json --old-plan '+c.remote+'/evals/fixtures/venus-control-corpus-v4-20261026.json --old-timelines '+c.OUT+'/timelines-v7-repaired --old-approval '+c.OUT+'/data-approval-v7.json --voice-reference '+c.OUT+'/george-reference-v33/reference-16k.wav --voice-reference-sha256 40159a59a35053c358d5192b3919a86453c8c6bf1b2019b289119ceca444f674'
        for arm in ('main','control'):
            execute('duplex-large65-native-'+arm,'training.venus.large_behavior',evaluate+' --arm '+arm,int(end-time.time()-600))
        execute('duplex-large65-combine','training.venus.large_behavior',' --combine --root '+OUTPUT+' --source '+source,120)
        c.record('complete64_native_speech_collection',source=source,adopted=False)
    except BaseException as error:
        c.record('capacity_interval_failure',error_type=type(error).__name__,error=str(error)[-1500:]);raise
    finally:
        if c.ssh_argv:
            try:
                c.ssh('mkdir -p '+OUTPUT+'/journals')
                for unit in c.UNITS:
                    state=c.props(unit)
                    if state['ActiveState']=='active' and state['SubState']!='exited':c.ssh('sudo systemctl stop '+unit)
                    c.ssh('sudo journalctl -u '+unit+' --no-pager -o cat > '+OUTPUT+'/journals/'+unit+'.log')
                copyback();c.record('all_native_outputs_offmachine_hash_verified',backup=str(BACKUP))
            except Exception as error:c.record('copyback_incomplete_unique_disk_preserved',error_type=type(error).__name__,error=str(error)[-1000:])
        if created:
            actual=[v for v in c.cloud('compute','instances','list') if v['name']==NAME]
            if actual:
                if not actual[0].get('lastStartTimestamp'):
                    save('cancelled-before-start.json',actual[0])
                    c.run(['gcloud','compute','instances','delete',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
                elif actual[0]['status']!='TERMINATED':
                    c.run(['gcloud','compute','instances','stop',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet'],150)
                if actual[0].get('lastStartTimestamp'):
                    final=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);save('provider-final.json',final)
                    elapsed=(datetime.fromisoformat(final['lastStopTimestamp'])-datetime.fromisoformat(final['lastStartTimestamp'])).total_seconds()
                    c.record('actual_GPU_stopped',vm_id=final['id'],actual_seconds=elapsed,upper_seconds=math.ceil(elapsed/100)*100)
        if watch:
            (R/'stop-requested').write_text('closed\n')
            try:watch.wait(timeout=35)
            except subprocess.TimeoutExpired:watch.terminate()
        awake.terminate()


if __name__=='__main__':main()
