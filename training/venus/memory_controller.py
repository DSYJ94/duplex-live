"""Sequential STANDARD replacement on the SAME staged disk, within remaining cap."""
import hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from . import controller as c
from .memory_guard import AUTH
from portable.context.gcp_budget_watchdog import read_runpod_pods
BASE=Path('runs/venus-training-20261026');OLD=BASE/'sustain-v10'

def main():
    c.R=BASE/'memory-v11';assert not c.R.exists(),'Observe the real operation; never duplicate'
    assert not c.run(['git','status','--porcelain','--','training/venus','data/venus','portable/venus/guard.py']).strip()
    grant=json.loads(AUTH.read_text());r=grant['resources'];zone=r['zone'];c.NAME=r['new_name']
    vms=c.cloud('compute','instances','list');assert all(v['status']=='TERMINATED' for v in vms)
    old=next(v for v in vms if str(v['id'])==r['retire_only_vm_id']);assert old['name']==r['retire_only_vm_name']
    last=json.loads((OLD/'provider-final.json').read_text());assert old['lastStopTimestamp']==last['lastStopTimestamp']
    assert 6400+datetime.fromisoformat(last['lastStopTimestamp']).timestamp()-datetime.fromisoformat(last['lastStartTimestamp']).timestamp()<=9500
    # Full independent local inventory check before retiring ONLY our stopped VM.
    rows=json.loads((OLD/'unique-inventory.json').read_text());assert len(rows)==3311 and sum(x['bytes'] for x in rows)==1256325325
    for row in rows:
        p=OLD/'copyback'/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==row['sha256']
    disk=c.cloud('compute','disks','describe',r['same_disk_name'],'--zone='+zone);assert str(disk['id'])==r['same_disk_id'] and int(disk['sizeGb'])==100
    pods=read_runpod_pods(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
    assert all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    price=json.loads(Path('runs/venus-20261026/us-current-price40-v1.json').read_text());assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    c.R.mkdir();c.source=c.run(['git','rev-parse','HEAD']).strip();c.remote=c.ROOT+'/duplex-venus-memory-'+c.source[:12]
    c.run(['git','bundle','create',str(c.R/'source.bundle'),'HEAD'],120)
    c.record('preflight_full_unique_backup_reverified',files=3311,bytes=1256325325,old_vm=old['id'],same_disk=disk['id'],prior_new_gpu_seconds_upper=9500,next_max_seconds=4800,hourly_price=price['hourly_estimate_usd'])
    awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())]);guard=None;created=False;vm=None
    try:
        c.run(['gcloud','compute','instances','delete',old['name'],'--zone='+zone,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
        disk=c.cloud('compute','disks','describe',r['same_disk_name'],'--zone='+zone);assert str(disk['id'])==r['same_disk_id'] and not disk.get('users')
        c.record('only_old_vm_retired_same_disk_preserved',old_vm_id=old['id'],disk_id=disk['id'])
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=2h')
        startup=c.R/'bootstrap.sh';startup.write_text('#!/bin/sh\nset -eu\nmkdir -p '+c.ROOT+'\nchmod 1777 '+c.ROOT+'\nsystemd-run --unit=duplex-venus-memory-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        operation=c.cloud('compute','instances','create',c.NAME,'--zone='+zone,'--machine-type=a2-highgpu-1g','--provisioning-model=STANDARD','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+r['same_disk_name']+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(startup),'--max-run-duration=4800s','--instance-termination-action=STOP','--async');operation=operation[0] if isinstance(operation,list) else operation;created=True
        c.record('one_standard_replacement_submitted',operation=operation['name'],source=c.source,same_disk_id=disk['id'])
        until=time.time()+240
        while operation['status']!='DONE':
            if time.time()>until:raise TimeoutError('Observe same operation; no duplicate')
            time.sleep(3);operation=c.cloud('compute','operations','describe',operation['name'],'--zone='+zone)
        if operation.get('error'):raise RuntimeError(json.dumps(operation['error']))
        vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);assert vm['status']=='RUNNING' and str(vm['id'])!=old['id']
        start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+4800
        assert int(vm['scheduling']['maxRunDuration']['seconds'])==4800
        c.record('actual_same_disk_standard_running',vm_id=vm['id'],disk_id=disk['id'],last_start=vm['lastStartTimestamp'],native_stop_unix=end)
        for _ in range(30):
            try:keys=c.cloud('compute','instances','get-guest-attributes',c.NAME,'--zone='+zone,'--query-path=hostkeys/')
            except RuntimeError:keys=[]
            if keys:break
            time.sleep(3)
        else:raise RuntimeError('Google hostkeys unavailable')
        private=c.R/'.private';private.mkdir(mode=0o700);known=private/'known-hosts';alias='gcp-venus-memory-v11';known.write_text(''.join(alias+' '+k['key']+' '+k['value']+'\n' for k in keys));known.chmod(0o600)
        proxy=shlex.join(['gcloud','compute','start-iap-tunnel',c.NAME,'22','--listen-on-stdin','--zone='+zone,'--project='+c.PROJECT,'--verbosity=error'])
        c.ssh_argv=['ssh','-i','runs/local-migration-20261024/.private/gcp-native-ssh','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias='+alias,'-o','UserKnownHostsFile='+str(known),'-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ControlMaster=auto','-o','ControlPersist=180','-o','ControlPath=/private/tmp/duplex-venus-memory-v11','-o','ProxyCommand='+proxy,'daryl_aurium_labs_com@'+vm['networkInterfaces'][0]['accessConfigs'][0]['natIP']]
        (c.R/'ssh.json').write_text(json.dumps(c.ssh_argv)+'\n')
        for _ in range(15):
            try:c.ssh('test -d '+c.OUT+'/native-cache-v8 && nvidia-smi',20);break
            except RuntimeError:time.sleep(3)
        else:raise RuntimeError('Strict IAP SSH not ready')
        c.sync(c.R/'source.bundle',c.ROOT+'/memory-source-v11.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/memory-source-v11.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+c.source)
        lease=dict(resource_key='venus-memory-v11',zone=zone,vm_id=str(vm['id']),disk_id=str(disk['id']),disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=9500,prior_gpu_seconds=40351,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=hashlib.sha256(AUTH.read_bytes()).hexdigest(),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        (c.R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');c.sync(c.R/'lease.json',c.ROOT+'/venus-memory-lease-v11.json')
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(c.R/'lease.json'),'--ssh-argv',str(c.R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(c.R/'guard.json'),'--stop-file',str(c.R/'stop-requested')],stdout=(c.R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});c.guard=guard
        until=time.time()+90
        while not (c.R/'guard.json').exists():
            if guard.poll() is not None:raise RuntimeError('Independent Mac guard ended')
            if time.time()>until:raise TimeoutError('No new verified guard')
            time.sleep(1)
        c.ssh('sudo systemd-run --unit=duplex-venus-memory-guest-guard-v11 --property=RuntimeMaxSec=4900 --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+c.ROOT+'/venus-memory-lease-v11.json')
        assert c.props('duplex-venus-memory-guest-guard-v11')['ActiveState']=='active'
        c.ssh('sudo systemctl stop duplex-venus-memory-bootstrap-stop.timer');c.record('both_guards_live',mac_pid=guard.pid)
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python';env='CUBLAS_WORKSPACE_CONFIG=:4096:8 DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+c.ROOT+'/venus-memory-lease-v11.json HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache'
        def python(cmd):return '/usr/bin/env '+env+' '+py+' -u -m '+cmd
        prefix='duplex-venus-prefix-cache-v11';c.start_unit(prefix,python('training.venus.prefix_cache --model '+c.MODEL+' --cache '+c.OUT+'/native-cache-v8 --output '+c.OUT+'/prefix-cache-v11 --source '+c.source),max(1,int(end-time.time()-600)));c.wait_unit(prefix,end-600)
        args=' --model '+c.MODEL+' --cache '+c.OUT+'/prefix-cache-v11 --approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/candidate-v11 --source '+c.source+' --cache-producer '+c.source+' --lease '+c.ROOT+'/venus-memory-lease-v11.json'
        first='duplex-venus-suffix-numerical-start-v11';c.start_unit(first,python('training.venus.train --stage start'+args),max(1,int(end-time.time()-300)));c.wait_unit(first,end-300)
        recipe=json.loads(c.ssh('cat '+c.OUT+'/candidate-v11/recipe.json'));c.record('actual_schedule_admitted',recipe=recipe['recipe'],admission=recipe['admission'])
        sustained='duplex-venus-suffix-restore-sustain-v11';c.start_unit(sustained,python('training.venus.train --stage restore-continue'+args),max(1,int(end-time.time()-300)));c.wait_unit(sustained,end-300)
        terminal=json.loads(c.ssh('cat '+c.OUT+'/candidate-v11/training-terminal.json'));assert terminal['completed'] and not terminal['adopted']
        c.record('actual_training_terminal',result=terminal)
    except BaseException as error:c.record('memory_continuation_failure',error_type=type(error).__name__,error=str(error)[-2500:]);raise
    finally:
        if c.ssh_argv:
            try:
                for unit in c.UNITS:
                    state=c.props(unit)
                    if state['ActiveState']=='active' and state['SubState']!='exited':c.ssh('sudo systemctl stop '+unit,30)
                c.copyback()
            except Exception as error:c.record('copyback_error_disk_preserved',error=str(error)[-1500:])
        if guard and guard.poll() is None:
            (c.R/'stop-requested').write_text('closing\n')
            try:guard.wait(timeout=150)
            except subprocess.TimeoutExpired:c.record('guard_stop_observation_timeout',pid=guard.pid)
        if created:
            current=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone)
            if vm:assert str(current['id'])==str(vm['id'])
            if current['status'] not in ('TERMINATED','STOPPED'):c.run(['gcloud','compute','instances','stop',c.NAME,'--zone='+zone,'--project='+c.PROJECT,'--quiet'],150)
            current=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);(c.R/'provider-final.json').write_text(json.dumps(current,indent=2)+'\n');c.record('provider_stopped',vm_id=current['id'],status=current['status'])
        awake.terminate()

if __name__=='__main__':main()
