"""One conditional native training allocation. No duplicate submission on timeout.

Mac orchestrates the independent reviewer; only authored synthetic artifacts and
Git source reach the isolated GPU. Every terminal, partial result and checkpoint
is copied and hashed before resource retirement. Native STOP + two guards apply.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime,timezone
from .guard import AUTH
from portable.venus.guard import PROJECT,PRIOR_SECONDS,PRIOR_USD
from portable.context.gcp_budget_watchdog import read_runpod_pods

R=Path('runs/venus-training-20261026/allocation-v1')
NAME='duplex-venus-train-20261026-v1';TAG='duplex-venus-train-bounded'
ROOT='/workspace/migration';OUT=ROOT+'/venus-training-v1'
MODEL=ROOT+'/venus-public-v1/Realtime-Venus-Audio'
UNITS=[];events=[];vm=None;guard=None;ssh_argv=None;created=False;source=None
FIREWALL=['duplex-venus-train-iap-v1','duplex-venus-train-deny-v1']


def run(argv,timeout=60):
    p=subprocess.run(argv,text=True,capture_output=True,timeout=timeout)
    if p.returncode:raise RuntimeError(p.stderr[-2400:])
    return p.stdout


def cloud(*args):return json.loads(run(['gcloud',*args,'--project='+PROJECT,'--quiet','--format=json']))


def record(event,**details):
    row=dict(at_utc=datetime.now(timezone.utc).isoformat(),event=event,**details);events.append(row)
    (R/'events.json').write_text(json.dumps(events,indent=2)+'\n');print(json.dumps(row),flush=True)


def ssh(cmd,timeout=60):return run(ssh_argv+[cmd],timeout)


def sync(local,dest,timeout=180):
    return run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=core','--exclude=core.*','-e',shlex.join(ssh_argv[:-1]),str(local),ssh_argv[-1]+':'+dest],timeout)


def props(unit):
    return dict(x.split('=',1) for x in ssh('sudo systemctl show '+unit+' -p ActiveState -p SubState -p Result -p ExecMainStatus -p MainPID -p InvocationID').splitlines())


def start_unit(unit,cmd,seconds):
    if unit in UNITS:raise ValueError('One actual launch per named stage')
    UNITS.append(unit)
    ssh('sudo systemd-run --unit='+unit+' --uid=daryl_aurium_labs_com --working-directory='+remote+' --property=RemainAfterExit=yes --property=RuntimeMaxSec='+str(seconds)+' /bin/bash -c '+shlex.quote(cmd))
    record('actual_unit_started',unit=unit,state=props(unit))


def wait_unit(unit,deadline):
    while True:
        state=props(unit)
        if state['SubState']=='exited' or state['ActiveState'] in ('failed','inactive'):
            (R/(unit+'-terminal.json')).write_text(json.dumps(state,indent=2)+'\n')
            record('actual_unit_terminal',unit=unit,state=state)
            if state['Result']!='success' or state['ExecMainStatus']!='0':
                raise RuntimeError('Actual stage failed '+unit+': '+str(state)+'\n'+ssh('sudo journalctl -u '+unit+' --no-pager -o cat | tail -35',30))
            return state
        if guard.poll() is not None:raise RuntimeError('Independent provider guard ended')
        if time.time()>deadline:raise TimeoutError('Observe same stage; do not relaunch '+unit)
        time.sleep(5)


def copyback(transfer_timeout=240):
    ssh('mkdir -p '+OUT+'/journals; '+''.join('sudo journalctl -u '+u+' --no-pager -o cat > '+OUT+'/journals/'+u+'.log; ' for u in UNITS))
    code="""import hashlib,json
from pathlib import Path
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
r=Path('/workspace/migration/venus-training-v1')
rows=[dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(r.rglob('*')) if p.is_file() and not {'.private','credentials','owner-conversations','keys'}&set(p.relative_to(r).parts) and p.name not in ('session-token','core','API keys.md') and not p.name.startswith('core.')]
print(json.dumps(rows))"""
    rows=json.loads(ssh('/usr/bin/python3 -c '+shlex.quote(code),180));assert rows
    (R/'unique-inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
    dest=R/'copyback';dest.mkdir(exist_ok=True)
    run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=keys','--exclude=API keys.md','--exclude=core','--exclude=core.*','-e',shlex.join(ssh_argv[:-1]),ssh_argv[-1]+':'+OUT+'/',str(dest)+'/'],transfer_timeout)
    for row in rows:
        path=dest/row['path'];assert path.stat().st_size==row['bytes'] and hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==row['sha256']
    result=dict(all_hashes_verified=True,files=len(rows),bytes=sum(r['bytes'] for r in rows),source=source)
    (R/'copyback-verified.json').write_text(json.dumps(result,indent=2)+'\n');record('full_unique_copyback_verified',**result)


def main():
    global R,remote,vm,guard,ssh_argv,created,source
    zone=sys.argv[1] if len(sys.argv)>1 else 'us-central1-a'
    attempt=sys.argv[2] if len(sys.argv)>2 else 'v1'
    assert re.fullmatch(r'v[1-9][0-9]*',attempt)
    R=Path('runs/venus-training-20261026')/('allocation-'+attempt)
    assert not R.exists(),'Never duplicate a real allocation or reuse its lease'
    for prior in R.parent.glob('allocation-v*/events.json'):
        assert not any(e['event']=='one_actual_allocation_submitted' for e in json.loads(prior.read_text())), 'Prior allocation history must be carried in a revised authority before another allocation'
    R.mkdir(parents=True)
    auth=AUTH.read_bytes();grant=json.loads(auth);assert grant['status']=='APPROVED' and zone in grant['resources']['zones']
    source=run(['git','rev-parse','HEAD']).strip();remote=ROOT+'/duplex-venus-train-'+source[:12]
    # All required source must be committed; unrelated local changes are preserved.
    for folder in ('training/venus','data/venus'):
        assert not run(['git','status','--porcelain','--',folder]).strip()
    run(['git','bundle','create',str(R/'source.bundle'),'HEAD'],120)
    price=json.loads(Path('runs/venus-20261026/us-current-price40-v1.json').read_text())
    assert 0<price['hourly_estimate_usd']<=7.25 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    current=cloud('compute','instances','list');assert all(v['status'] in ('TERMINATED','STOPPED') for v in current)
    pods=read_runpod_pods(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
    assert all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    quota=cloud('compute','regions','describe','us-central1');q={x['metric']:x for x in quota['quotas']}
    assert q['NVIDIA_A100_GPUS']['limit']>=1 and q['NVIDIA_A100_GPUS']['usage']==0
    existing=cloud('compute','disks','list');assert not any(x['name']==NAME for x in existing)
    assert not any(x['name']==NAME for x in current)
    record('fresh_preflight_one_gpu_off',source=source,hourly_estimate_usd=price['hourly_estimate_usd'],prior_new_gpu_seconds_upper=4600,maximum_allocation_seconds=7200)
    awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())])
    try:
        assert not any(TAG in r.get('targetTags',[]) for r in cloud('compute','firewall-rules','list'))
        cloud('compute','firewall-rules','create',FIREWALL[0],'--network=default','--direction=INGRESS','--priority=800','--source-ranges=35.235.240.0/20','--target-tags='+TAG,'--allow=tcp:22')
        cloud('compute','firewall-rules','create',FIREWALL[1],'--network=default','--direction=INGRESS','--priority=900','--source-ranges=0.0.0.0/0','--target-tags='+TAG,'--action=DENY','--rules=all')
        cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=3h')
        startup=R/'bootstrap.sh';startup.write_text('#!/bin/sh\nset -eu\nmkdir -p '+ROOT+'\nchmod 1777 '+ROOT+'\nsystemd-run --unit=duplex-venus-train-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        operation=cloud('compute','instances','create',NAME,'--zone='+zone,'--machine-type=a2-highgpu-1g','--provisioning-model=FLEX_START','--request-valid-for-duration=900s','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags='+TAG,'--image=common-cu129-ubuntu-2204-nvidia-580-v20261019','--image-project=deeplearning-platform-release','--boot-disk-size=100GB','--boot-disk-type=pd-balanced','--boot-disk-device-name='+NAME,'--no-boot-disk-auto-delete','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(startup),'--max-run-duration=7200s','--instance-termination-action=STOP','--async')
        operation=operation[0] if isinstance(operation,list) else operation;created=True
        record('one_actual_allocation_submitted',operation=operation['name'],zone=zone)
        deadline=time.time()+960
        while operation['status']!='DONE':
            if time.time()>deadline:raise TimeoutError('Same allocation operation remains pending; no duplicate')
            time.sleep(5);operation=cloud('compute','operations','describe',operation['name'],'--zone='+zone)
        if operation.get('error'):raise RuntimeError(json.dumps(operation['error']))
        vm=cloud('compute','instances','describe',NAME,'--zone='+zone)
        while vm['status']!='RUNNING':
            if time.time()>deadline:raise TimeoutError('Same allocation not RUNNING')
            time.sleep(5);vm=cloud('compute','instances','describe',NAME,'--zone='+zone)
        disk=cloud('compute','disks','describe',NAME,'--zone='+zone);assert int(disk['sizeGb'])==100
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+7200
        record('actual_gpu_running',vm_id=vm['id'],disk_id=disk['id'],native_stop_unix=end,last_start=vm['lastStartTimestamp'])
        for _ in range(30):
            try:keys=cloud('compute','instances','get-guest-attributes',NAME,'--zone='+zone,'--query-path=hostkeys/')
            except RuntimeError:keys=[]
            if keys:break
            time.sleep(3)
        else:raise RuntimeError('Google hostkey attributes unavailable')
        private=R/'.private';private.mkdir(mode=0o700);known=private/'known-hosts';alias='gcp-venus-train-v1';known.write_text(''.join(alias+' '+k['key']+' '+k['value']+'\n' for k in keys));known.chmod(0o600)
        proxy=shlex.join(['gcloud','compute','start-iap-tunnel',NAME,'22','--listen-on-stdin','--zone='+zone,'--project='+PROJECT,'--verbosity=error'])
        ssh_argv=['ssh','-i','runs/local-migration-20261024/.private/gcp-native-ssh','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias='+alias,'-o','UserKnownHostsFile='+str(known),'-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ControlMaster=auto','-o','ControlPersist=180','-o','ControlPath=/private/tmp/duplex-venus-train-v1-mux','-o','ProxyCommand='+proxy,'daryl_aurium_labs_com@'+vm['networkInterfaces'][0]['accessConfigs'][0]['natIP']]
        (R/'ssh.json').write_text(json.dumps(ssh_argv)+'\n')
        for _ in range(12):
            try:record('actual_strict_ssh_gpu',stdout=ssh('test -d '+ROOT+' && nvidia-smi',20));break
            except RuntimeError:time.sleep(3)
        else:raise RuntimeError('Strict IAP SSH not ready')
        sync(R/'source.bundle',ROOT+'/source.bundle')
        ssh('git init --bare '+ROOT+'/source.git && git --git-dir='+ROOT+'/source.git fetch '+ROOT+'/source.bundle HEAD && git --git-dir='+ROOT+'/source.git worktree add --detach '+remote+' '+source)
        lease=dict(resource_key='venus-training-v1',zone=zone,vm_id=str(vm['id']),disk_id=str(disk['id']),disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=4600,prior_gpu_seconds=PRIOR_SECONDS,prior_scope_usd=PRIOR_USD,authorization_sha256=hashlib.sha256(auth).hexdigest(),controller_pid=os.getpid(),controller_birth=run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        (R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');sync(R/'lease.json',ROOT+'/venus-training-lease.json')
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(R/'lease.json'),'--ssh-argv',str(R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(R/'guard.json'),'--stop-file',str(R/'stop-requested')],stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        until=time.time()+90
        while not (R/'guard.json').exists():
            if guard.poll() is not None:raise RuntimeError('Mac guard ended')
            if time.time()>until:raise TimeoutError('No verified live guard')
            time.sleep(1)
        ssh('sudo systemd-run --unit=duplex-venus-train-guest-guard-v1 --property=RuntimeMaxSec=7300 --working-directory='+remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+ROOT+'/venus-live-guard-v1.json --lease '+ROOT+'/venus-training-lease.json')
        assert props('duplex-venus-train-guest-guard-v1')['ActiveState']=='active'
        ssh('sudo systemctl stop duplex-venus-train-bootstrap-stop.timer');record('both_guards_live',guard_pid=guard.pid)
        ssh('mkdir -p '+OUT+' '+ROOT+'/venus-runtime-v1 '+ROOT+'/venus-public-v1')
        constraints=remote+'/portable/venus/baseline-runtime-constraints-20261026.txt'
        setup='set -eu; sudo apt-get update -qq; sudo apt-get install -y -qq python3-venv libsndfile1; /usr/bin/python3 -m venv '+ROOT+'/venus-runtime-v1/venv; '+ROOT+'/venus-runtime-v1/venv/bin/pip install --disable-pip-version-check --no-cache-dir -r '+remote+'/portable/venus/requirements.txt -c '+constraints+'; '+ROOT+'/venus-runtime-v1/venv/bin/pip freeze > '+OUT+'/pip-freeze.txt'
        start_unit('duplex-venus-train-setup-v1',setup,1500)
        sync('runs/venus-20261026/public-model-manifest.json',ROOT+'/public-model-manifest.json')
        start_unit('duplex-venus-train-fetch-v1','/usr/bin/python3 -u -m portable.venus.fetch_public --manifest '+ROOT+'/public-model-manifest.json --output '+ROOT+'/venus-public-v1 > '+OUT+'/public-download.log 2>&1',1000)
        sync('runs/venus-training-20261026/source-audio-v3/',OUT+'/source-audio-v3/',120)
        sync('runs/venus-training-20261026/timelines-v4-bound/',OUT+'/timelines-v4-bound/',120)
        wait_unit('duplex-venus-train-fetch-v1',min(end-300,time.time()+1050));wait_unit('duplex-venus-train-setup-v1',min(end-300,time.time()+1500))
        check="from pathlib import Path;assert set(Path('/workspace/migration/venus-training-v1/pip-freeze.txt').read_text().splitlines())==set(Path("+repr(constraints)+").read_text().splitlines());import torch;assert torch.cuda.is_available() and torch.cuda.device_count()==1;print('Exact baseline runtime and one GPU verified')"
        record('runtime_exact_match',stdout=ssh(ROOT+'/venus-runtime-v1/venv/bin/python -c '+shlex.quote(check)))
        py=ROOT+'/venus-runtime-v1/venv/bin/python'
        env='DUPLEX_LIVE_GUARD_FILE='+ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+ROOT+'/venus-training-lease.json HF_HOME='+ROOT+'/venus-runtime-v1/hf-cache'
        def python(cmd):return '/usr/bin/env '+env+' '+py+' -u -m '+cmd
        start_unit('duplex-venus-train-asr-v1',python('data.venus.asr --audio '+OUT+'/source-audio-v3 --output '+OUT+'/word-quality-v1'),900)
        wait_unit('duplex-venus-train-asr-v1',min(end-300,time.time()+950))
        # Small bounded report copy for the Mac-only independent reviewer.
        word=R/'word-quality.json';word.write_text(ssh('cat '+OUT+'/word-quality-v1/quality.json',60))
        audit='runs/venus-training-20261026/source-audit-v4.json';timelines='runs/venus-training-20261026/timelines-v4-bound';prepared=R/'prepared-semantic-v1.json'
        run([sys.executable,'-m','data.venus.semantic','prepare','--timelines',timelines,'--word-quality',str(word),'--audit',audit,'--output',str(prepared)])
        start_unit('duplex-venus-train-native-proof-v1',python('training.venus.native_cache --model '+MODEL+' --timelines '+OUT+'/timelines-v4-bound --output '+OUT+'/native-proof-v1 --proof-only'),600)
        # In parallel with actual native replay, never transfer the API credential.
        reviewlog=(R/'semantic-review.log').open('x')
        review=subprocess.Popen([sys.executable,'-u','-m','data.venus.semantic','execute','--prepared',str(prepared),'--output',str(R/'semantic-v1')],stdout=reviewlog,stderr=subprocess.STDOUT)
        wait_unit('duplex-venus-train-native-proof-v1',min(end-300,time.time()+650))
        until=time.time()+300
        while review.poll() is None:
            if time.time()>until:review.terminate();raise TimeoutError('Independent review blocked; stop paid compute')
            if guard.poll() is not None:review.terminate();raise RuntimeError('Guard ended during reviewer')
            time.sleep(3)
        reviewlog.close();assert review.returncode==0,'Independent complete semantic review failed'
        approval=R/'data-approval.json'
        run([sys.executable,'-m','data.venus.semantic','approve','--timelines',timelines,'--word-quality',str(word),'--audit',audit,'--prepared',str(prepared),'--semantic-quality',str(R/'semantic-v1/semantic-quality.json'),'--output',str(approval)])
        sync(approval,OUT+'/data-approval.json');record('complete_data_approved',approval_sha256=hashlib.sha256(approval.read_bytes()).hexdigest())
        start_unit('duplex-venus-train-cache-v1',python('training.venus.native_cache --model '+MODEL+' --timelines '+OUT+'/timelines-v4-bound --approval '+OUT+'/data-approval.json --output '+OUT+'/native-cache-v1'),max(1,int(end-time.time()-300)))
        wait_unit('duplex-venus-train-cache-v1',end-300)
        args=' --model '+MODEL+' --cache '+OUT+'/native-cache-v1 --approval '+OUT+'/data-approval.json --output '+OUT+'/candidate-v1 --source '+source+' --lease '+ROOT+'/venus-training-lease.json'
        start_unit('duplex-venus-train-numerical-start-v1',python('training.venus.train --stage start'+args),max(1,int(end-time.time()-300)))
        wait_unit('duplex-venus-train-numerical-start-v1',end-300)
        start_unit('duplex-venus-train-restore-sustain-v1',python('training.venus.train --stage restore-continue'+args),max(1,int(end-time.time()-300)))
        wait_unit('duplex-venus-train-restore-sustain-v1',end-300)
        copyback();record('training_completed_behavior_still_required',candidate_adopted=False)
    except BaseException as error:
        record('controller_failure',error_type=type(error).__name__,error=str(error)[-2500:])
        if ssh_argv:
            try:
                for unit in UNITS:
                    state=props(unit)
                    if state['ActiveState']=='active' and state['SubState']!='exited':ssh('sudo systemctl stop '+unit,30)
                copyback()
            except Exception as partial:record('partial_copyback_failure_disk_preserved',error=str(partial)[-1200:])
        raise
    finally:
        if guard and guard.poll() is None:
            (R/'stop-requested').write_text('closing\n')
            try:guard.wait(timeout=150)
            except subprocess.TimeoutExpired:record('guard_stop_observation_timeout',pid=guard.pid)
        if created:
            try:
                current=cloud('compute','instances','describe',NAME,'--zone='+zone)
                if vm:assert str(current['id'])==str(vm['id'])
                if current['status'] not in ('TERMINATED','STOPPED'):run(['gcloud','compute','instances','stop',NAME,'--zone='+zone,'--project='+PROJECT,'--quiet'],150)
                current=cloud('compute','instances','describe',NAME,'--zone='+zone);(R/'provider-final.json').write_text(json.dumps(current,indent=2)+'\n');record('provider_stopped',vm_id=current['id'],status=current['status'])
            except Exception as error:record('provider_cleanup_observation_error',error=str(error)[-1000:])
        awake.terminate()


if __name__=='__main__':main()
