"""One approved action-fidelity continuation; same disk, no duplicate allocation."""
import json,os,shutil,subprocess,sys,time,shlex
from pathlib import Path
from datetime import datetime
from . import controller as c
from .runtime_guard import AUTH,PLAN,KEY,RETRY_PLAN,RETRY_KEY,sha as digest
from portable.venus.guard import bounded_runpod_read


def main():
    global PLAN,KEY
    retry=False
    if retry:PLAN,KEY=RETRY_PLAN,RETRY_KEY
    version='v44';unit_prefix='duplex-venus-action'+version[1:]
    c.R=Path('runs/venus-training-20261029')/('action-'+version)
    assert not c.R.exists(),'Observe existing action-v41; never duplicate its allocation'
    a=json.loads(AUTH.read_text());p=json.loads(PLAN.read_text());zone='us-central1-a'
    assert a['status']=='APPROVED' and a['human_approval']=='Approve 108 updates within existing time/$70 cap'
    assert p['parent_sha256']==digest(AUTH) and a['maximum_candidate_new_updates']==108
    c.NAME=p['vm_name'];duration=p['maximum_gpu_seconds'];prior=p['focused_gpu_seconds_upper_before_launch']
    r=dict(same_existing_disk_name=p['same_disk_name'],same_existing_disk_id=a['same_disk_id'])
    assert not c.run(['git','status','--porcelain','--','training/venus','portable/venus','docs/NATIVE-VENUS-ACTION-FIDELITY-AUTHORIZATION-2026-10-29.json','docs/NATIVE-VENUS-ACTION-FIDELITY-EXECUTION-2026-10-29.json']).strip()
    seed=Path('runs/venus-training-20261029/action-v43');before=Path('runs/venus-training-20261029/foundation43-artifact-recovery-v2')
    last=json.loads((before/'provider-final.json').read_text());old=c.cloud('compute','instances','describe',last['name'],'--zone='+zone)
    assert old['status']=='TERMINATED' and old['id']==last['id']==p['retire_only_vm_id'] and old['lastStopTimestamp']==last['lastStopTimestamp']
    assert all(v['status']=='TERMINATED' for v in c.cloud('compute','instances','list'))
    quotas={q['metric']:q for q in c.cloud('compute','regions','describe','us-central1')['quotas']}
    assert quotas['NVIDIA_A100_GPUS']['usage']==0 and quotas['NVIDIA_A100_GPUS']['limit']>=1
    disk=c.cloud('compute','disks','describe',r['same_existing_disk_name'],'--zone='+zone)
    assert disk['id']==r['same_existing_disk_id'] and int(disk['sizeGb'])==100
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
    assert all(x['status'] in ('EXITED','STOPPED','TERMINATED') for x in pods)
    price=json.loads(Path('runs/venus-20261026/us-current-price40-v1.json').read_text());assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    rows=json.loads((seed/'unique-inventory.json').read_text());backup=json.loads((before/'copyback-verified.json').read_text())
    assert backup['all_hashes_verified'] and len(rows)==17473 and sum(v['bytes'] for v in rows)==8970439147
    # Already full-SHA-verified backup, unchanged. Recheck inventory/file sizes and
    # critical state; don't spend GPU time rehashing old artifacts before training.
    assert all((seed/'copyback'/row['path']).stat().st_size==row['bytes'] for row in rows)
    assert digest(seed/'copyback/adaptive-candidate-v34/step-0054.pt')==a['parent_checkpoint_sha256']
    c.R.mkdir(parents=True);c.source=c.run(['git','rev-parse','HEAD']).strip();c.remote=c.ROOT+'/duplex-venus-action-'+c.source[:12]
    c.run(['git','bundle','create',str(c.R/'source.bundle'),'HEAD'],120)
    c.run(['git','-C','/private/tmp/duplex-opal-upstream-20261028','bundle','create',str((c.R/'opal-source.bundle').resolve()),'HEAD'],60)
    c.record('adaptive_preflight',source=c.source,prior_focused_seconds=prior,maximum_next_seconds=duration,backup=backup,price_usd_hour=price['hourly_estimate_usd'],external_calls=0)
    guard=None;created=False;vm=None;awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())])
    try:
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=4h')
        c.run(['gcloud','compute','instances','delete',old['name'],'--zone='+zone,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
        c.record('only_stopped_envelope_retired_same_disk_contents_preserved',old_id=old['id'],disk_id=disk['id'])
        current_disk=c.cloud('compute','disks','describe',r['same_existing_disk_name'],'--zone='+zone);assert current_disk['id']==disk['id'] and not current_disk.get('users')
        bootstrap=c.R/'bootstrap.sh';bootstrap.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit='+unit_prefix+'-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        op=c.cloud('compute','instances','create',c.NAME,'--zone='+zone,'--machine-type=a2-highgpu-1g','--provisioning-model=FLEX_START','--request-valid-for-duration=900s','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+r['same_existing_disk_name']+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(bootstrap),'--max-run-duration='+str(duration)+'s','--instance-termination-action=STOP','--async');op=op[0] if isinstance(op,list) else op;created=True
        c.record('one_adaptive_allocation',operation=op['name'],source=c.source)
        until=time.time()+960
        while op['status']!='DONE':
            if time.time()>until:raise TimeoutError('Observe the same pending operation; do not resubmit')
            time.sleep(5);op=c.cloud('compute','operations','describe',op['name'],'--zone='+zone)
        if op.get('error'):raise RuntimeError(json.dumps(op['error']))
        vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);assert vm['status']=='RUNNING'
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+duration
        c.record('actual_adaptive_running',vm_id=vm['id'],same_disk_id=disk['id'],last_start=vm['lastStartTimestamp'],native_stop_unix=end)
        for _ in range(30):
            try:keys=c.cloud('compute','instances','get-guest-attributes',c.NAME,'--zone='+zone,'--query-path=hostkeys/')
            except RuntimeError:keys=[]
            if keys:break
            time.sleep(3)
        else:raise RuntimeError('Authenticated host keys unavailable')
        private=c.R/'.private';private.mkdir(mode=0o700);known=private/'known-hosts';alias='gcp-venus-adaptive-'+version
        known.write_text(''.join(alias+' '+k['key']+' '+k['value']+'\n' for k in keys));known.chmod(0o600)
        c.ssh_argv=json.loads((seed/'ssh.json').read_text());c.ssh_argv=['HostKeyAlias='+alias if x.startswith('HostKeyAlias=') else 'UserKnownHostsFile='+str(known) if x.startswith('UserKnownHostsFile=') else ('ControlPath=/private/tmp/duplex-venus-adaptive-'+version if x.startswith('ControlPath=') else x).replace('start-iap-tunnel '+old['name']+' ','start-iap-tunnel '+c.NAME+' ') for x in c.ssh_argv]
        c.ssh_argv[-1]=c.ssh_argv[-1].split('@',1)[0]+'@'+vm['networkInterfaces'][0]['accessConfigs'][0]['natIP'];(c.R/'ssh.json').write_text(json.dumps(c.ssh_argv)+'\n')
        for _ in range(20):
            try:c.ssh('nvidia-smi',20);break
            except RuntimeError:time.sleep(3)
        else:raise RuntimeError('Strict SSH not ready')
        bundle=c.ROOT+'/adaptive-v34-source.bundle';lease_path=c.ROOT+'/action-'+version+'-lease.json';c.sync(c.R/'source.bundle',bundle)
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+bundle+' HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+c.source)
        opal_bundle=c.ROOT+'/adaptive-v34-opal.bundle';c.sync(c.R/'opal-source.bundle',opal_bundle)
        c.ssh('git init '+c.ROOT+'/opal-upstream-v34 && git -C '+c.ROOT+'/opal-upstream-v34 fetch '+opal_bundle+' HEAD && git -C '+c.ROOT+'/opal-upstream-v34 checkout --detach 8b16759a3e9930916ed582e28068267859f935f8')
        assert c.ssh('git -C '+c.ROOT+'/opal-upstream-v34 rev-parse HEAD').strip()=='8b16759a3e9930916ed582e28068267859f935f8'
        lease=dict(resource_key=KEY,zone=zone,vm_id=vm['id'],disk_id=disk['id'],disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=prior,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=digest(AUTH),execution_plan_sha256=digest(PLAN),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        (c.R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');c.sync(c.R/'lease.json',lease_path)
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(c.R/'lease.json'),'--ssh-argv',str(c.R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(c.R/'guard.json'),'--stop-file',str(c.R/'stop-requested')],stdout=(c.R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});c.guard=guard
        until=time.time()+90
        while not (c.R/'guard.json').exists():
            if guard.poll() is not None:raise RuntimeError('Independent Mac guard ended')
            if time.time()>until:raise TimeoutError('Guard readiness timeout')
            time.sleep(1)
        live=json.loads(c.ssh('cat '+c.ROOT+'/venus-live-guard-v1.json'));assert live['vm_id']==vm['id'] and time.time()<live['expires_unix'] and live['stop_unix']==end
        c.ssh('sudo systemd-run --unit='+unit_prefix+'-guest --property=RuntimeMaxSec=7300 --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lease_path)
        assert c.props(unit_prefix+'-guest')['ActiveState']=='active';c.ssh('sudo systemctl stop '+unit_prefix+'-bootstrap-stop.timer');c.record('both_guards_live',mac_pid=guard.pid)
        dest=c.R/'copyback';dest.mkdir()
        for row in rows:
            original=seed/'copyback'/row['path'];target=dest/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(original,target)
            except OSError:shutil.copy2(original,target)
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python';env='/usr/bin/env CUBLAS_WORKSPACE_CONFIG=:4096:8 DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+lease_path+' HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache '+py
        assert c.ssh('sha256sum '+c.OUT+'/adaptive-candidate-v34/step-0054.pt').split()[0]==a['parent_checkpoint_sha256']
        from .runtime_stage import execute
        execute(c,env,end,lease_path)
    except BaseException as error:c.record('action_failure',error_type=type(error).__name__,error=str(error)[-2000:]);raise
    finally:
        if c.ssh_argv and c.UNITS:
            try:
                for unit in c.UNITS:
                    state=c.props(unit)
                    if state['ActiveState']=='active' and state['SubState']!='exited':c.ssh('sudo systemctl stop '+unit,30)
                c.copyback(600)
            except Exception as error:c.record('copyback_failure_disk_preserved',error_type=type(error).__name__,error=str(error)[-500:])
        elif c.ssh_argv:
            c.record('no_model_units_started_prior_complete_backup_retained',backup=str(before/'copyback-verified.json'),new_optimizer_updates=0)
        if guard and guard.poll() is None:
            (c.R/'stop-requested').write_text('closing\n')
            try:guard.wait(timeout=150)
            except subprocess.TimeoutExpired:guard.terminate()
        if created:
            current=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone)
            if current['status'] not in ('TERMINATED','STOPPED'):c.run(['gcloud','compute','instances','stop',c.NAME,'--zone='+zone,'--project='+c.PROJECT,'--quiet','--discard-local-ssd=True'],150)
            final=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);(c.R/'provider-final.json').write_text(json.dumps(final,indent=2)+'\n');assert final['status']=='TERMINATED'
            elapsed=datetime.fromisoformat(final['lastStopTimestamp']).timestamp()-datetime.fromisoformat(final['lastStartTimestamp']).timestamp();upper=__import__('math').ceil(elapsed/100)*100
            c.record('provider_stopped',status=final['status'],actual_seconds=elapsed,focused_seconds_upper=prior+upper,prior_gpu_seconds=54451)
        awake.terminate()

if __name__=='__main__':main()
