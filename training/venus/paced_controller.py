"""One approved nonredundant paced continuation; no allocation during source review."""
import hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from . import controller as c
from .paced_guard import AUTH,PLAN
from portable.context.gcp_budget_watchdog import read_runpod_pods

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    c.R=Path('runs/venus-training-20261027/paced-continuation-v22');assert not c.R.exists(),'Observe actual controller, never duplicate'
    assert not c.run(['git','status','--porcelain','--','training/venus','data/venus','portable/venus','evals/harness/test_venus_training.py','evals/harness/test_venus_behavior.py']).strip()
    p=json.loads(PLAN.read_text());a=json.loads(AUTH.read_text());r=a['resources'];zone=r['zone'];c.NAME=p['vm_name']
    assert p['status']=='WITHIN_APPROVED_FOCUSED_REPAIR' and p['parent_sha256']==digest(AUTH)
    for key in ('approval','manifest','source_audit','semantic_quality','prepared_semantic','word_quality'):
        assert digest(p[key])==p[key+'_sha256']
    approval=json.loads(Path(p['approval']).read_text());assert approval['approved'] and (approval['train'],approval['dev'],approval['holdout'])==(216,52,24)
    seed=Path('runs/venus-training-20261027/cadence-qualify-v21');last=json.loads((seed/'provider-final.json').read_text())
    assert json.loads((seed/'copyback-verified.json').read_text())['all_hashes_verified']
    vms=c.cloud('compute','instances','list');assert all(v['status']=='TERMINATED' for v in vms)
    old=next(v for v in vms if str(v['id'])==p['retire_only_vm_id']);assert old['name']==p['retire_only_vm_name'] and old['lastStopTimestamp']==last['lastStopTimestamp']
    disk=c.cloud('compute','disks','describe',r['same_existing_disk_name'],'--zone='+zone);assert str(disk['id'])==r['same_existing_disk_id'] and int(disk['sizeGb'])==100
    rows=json.loads((seed/'unique-inventory.json').read_text());assert len(rows)==6117 and sum(x['bytes'] for x in rows)==3925806584
    for row in rows:
        file=seed/'copyback'/row['path'];assert file.stat().st_size==row['bytes'] and digest(file)==row['sha256']
    pods=read_runpod_pods(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods;assert all(x['status'] in ('EXITED','STOPPED','TERMINATED') for x in pods)
    price=json.loads(Path('runs/venus-20261026/us-current-price40-v1.json').read_text());assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    quota={x['metric']:x for x in c.cloud('compute','regions','describe','us-central1')['quotas']};assert quota['NVIDIA_A100_GPUS']['limit']>=1 and quota['NVIDIA_A100_GPUS']['usage']==0
    c.R.mkdir();c.source=c.run(['git','rev-parse','HEAD']).strip();c.remote=c.ROOT+'/duplex-venus-paced-'+c.source[:12]
    c.run(['git','bundle','create',str(c.R/'source.bundle'),'HEAD'],120)
    c.record('full_source_and_backup_approved',source=c.source,files=len(rows),bytes=sum(x['bytes'] for x in rows),focused_prior_seconds=2400,next_max_seconds=10000,external_semantic_maximum_usd=p['semantic_maximum_cost_usd'])
    awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())]);guard=None;created=False;vm=None
    try:
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=4h')
        c.run(['gcloud','compute','instances','delete',old['name'],'--zone='+zone,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
        disk=c.cloud('compute','disks','describe',r['same_existing_disk_name'],'--zone='+zone);assert str(disk['id'])==r['same_existing_disk_id'] and not disk.get('users')
        c.record('only_stopped_envelope_retired_same_disk_preserved',old_vm_id=str(old['id']),disk_id=str(disk['id']))
        startup=c.R/'bootstrap.sh';startup.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-venus-paced-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        op=c.cloud('compute','instances','create',c.NAME,'--zone='+zone,'--machine-type=a2-highgpu-1g','--provisioning-model=FLEX_START','--request-valid-for-duration=900s','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+r['same_existing_disk_name']+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(startup),'--max-run-duration=10000s','--instance-termination-action=STOP','--async');op=op[0] if isinstance(op,list) else op;created=True
        c.record('one_actual_flex_allocation',operation=op['name'],same_disk_id=str(disk['id']),maximum_seconds=10000)
        until=time.time()+960
        while op['status']!='DONE':
            if time.time()>until:raise TimeoutError('Observe same allocation operation, never resubmit')
            time.sleep(5);op=c.cloud('compute','operations','describe',op['name'],'--zone='+zone)
        if op.get('error'):raise RuntimeError(json.dumps(op['error']))
        vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);assert vm['status']=='RUNNING'
        start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+10000
        assert int(vm['scheduling']['maxRunDuration']['seconds'])==10000
        c.record('actual_same_disk_a100_running',vm_id=str(vm['id']),disk_id=str(disk['id']),last_start=vm['lastStartTimestamp'],native_stop_unix=end)
        # Boot disk host keys are independently pinned in the preceding verified run.
        c.ssh_argv=json.loads((seed/'ssh.json').read_text())
        c.ssh_argv=[x.replace('ControlPath=/private/tmp/duplex-venus-v21','ControlPath=/private/tmp/duplex-venus-v22').replace('start-iap-tunnel '+old['name']+' ','start-iap-tunnel '+c.NAME+' ') for x in c.ssh_argv]
        c.ssh_argv[-1]='daryl_aurium_labs_com@'+vm['networkInterfaces'][0]['accessConfigs'][0]['natIP'];(c.R/'ssh.json').write_text(json.dumps(c.ssh_argv)+'\n')
        for _ in range(20):
            try:c.ssh('test -f '+c.OUT+'/candidate-v13/step-0190.pt && nvidia-smi',20);break
            except RuntimeError:time.sleep(3)
        else:raise RuntimeError('Strict IAP SSH not ready')
        bundle=c.ROOT+'/paced-v22-source.bundle';lease_path=c.ROOT+'/paced-v22-lease.json';c.sync(c.R/'source.bundle',bundle)
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+bundle+' HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+c.source)
        lease=dict(resource_key='venus-paced-continuation-v22',zone=zone,vm_id=str(vm['id']),disk_id=str(disk['id']),disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=2400,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=digest(AUTH),execution_plan_sha256=digest(PLAN),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        (c.R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');c.sync(c.R/'lease.json',lease_path)
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(c.R/'lease.json'),'--ssh-argv',str(c.R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(c.R/'guard.json'),'--stop-file',str(c.R/'stop-requested')],stdout=(c.R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});c.guard=guard
        until=time.time()+90
        while not (c.R/'guard.json').exists():
            if guard.poll() is not None:raise RuntimeError('Mac guard ended')
            if time.time()>until:raise TimeoutError('Independent guard not ready')
            time.sleep(1)
        live=json.loads(c.ssh('cat '+c.ROOT+'/venus-live-guard-v1.json'));assert live['vm_id']==str(vm['id']) and live['stop_unix']==end and time.time()<live['expires_unix']
        guest='duplex-venus-paced-guest-guard-v22';c.ssh('sudo systemd-run --unit='+guest+' --property=RuntimeMaxSec=10100 --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lease_path);assert c.props(guest)['ActiveState']=='active'
        c.ssh('sudo systemctl stop duplex-venus-paced-bootstrap-stop.timer');c.record('both_independent_guards_live',mac_pid=guard.pid)
        dest=c.R/'copyback';dest.mkdir()
        for row in rows:
            oldfile=seed/'copyback'/row['path'];target=dest/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(oldfile,target)
            except OSError:shutil.copy2(oldfile,target)
        for local,remote in ((p['timelines'],c.OUT+'/paced-timelines-v22'),(p['audio_root'],c.OUT+'/paced-source-inputs-v22')):c.sync(local+'/',remote+'/')
        c.sync(p['approval'],c.OUT+'/paced-data-approval-v22.json');c.sync(p['word_quality'],c.OUT+'/paced-word-quality-v22.json')
        env='CUBLAS_WORKSPACE_CONFIG=:4096:8 DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+lease_path+' HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache'
        def execute(label,args,reserve):
            cmd='/usr/bin/env '+env+' '+c.ROOT+'/venus-runtime-v1/venv/bin/python -u -m '+args
            unit='duplex-venus-paced-'+label+'-v22';c.start_unit(unit,cmd,max(1,int(end-time.time()-reserve)));c.wait_unit(unit,end-reserve)
        execute('native-cache','training.venus.native_cache --model '+c.MODEL+' --timelines '+c.OUT+'/paced-timelines-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --output '+c.OUT+'/paced-native-cache-v22',3300)
        execute('prefix-cache','training.venus.prefix_cache --model '+c.MODEL+' --cache '+c.OUT+'/paced-native-cache-v22 --output '+c.OUT+'/paced-prefix-cache-v22 --source '+c.source,3300)
        args=' --model '+c.MODEL+' --cache '+c.OUT+'/paced-prefix-cache-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --output '+c.OUT+'/paced-candidate-v22 --source '+c.source+' --cache-producer '+c.source+' --lease '+lease_path+' --parent-checkpoint '+c.OUT+'/candidate-v13/step-0190.pt --parent-sha256 '+p['parent_checkpoint_sha256']
        execute('numerical-start','training.venus.train --stage start'+args,3000)
        recipe=json.loads(c.ssh('cat '+c.OUT+'/paced-candidate-v22/recipe.json'));c.record('actual_complete_epoch_recipe_admitted',recipe=recipe['recipe'],admission=recipe['admission'])
        execute('restore-train','training.venus.train --stage restore-continue'+args,2800)
        terminal=json.loads(c.ssh('cat '+c.OUT+'/paced-candidate-v22/training-terminal.json'));assert terminal['completed'] and not terminal['adopted'];c.record('actual_training_complete',result=terminal)
        args='training.venus.paced_behavior --model '+c.MODEL+' --candidate '+c.OUT+'/paced-candidate-v22 --parent-checkpoint '+c.OUT+'/candidate-v13/step-0190.pt --parent-sha256 '+p['parent_checkpoint_sha256']+' --plan '+c.remote+'/evals/fixtures/venus-control-repair-source-v2-20261027.json --timelines '+c.OUT+'/paced-timelines-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --old-plan '+c.remote+'/evals/fixtures/venus-control-corpus-v4-20261026.json --old-timelines '+c.OUT+'/timelines-v7-repaired --old-approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/paced-native-paired-v22 --source '+c.source+' --lease '+lease_path
        execute('paired-native',args,600);c.record('complete_paired_native_audio_collected',fresh24_and_old8_each_condition=True,adopted=False,full_behavior_semantics_and_listening_required=True)
    except BaseException as error:c.record('paced_continuation_failure',error_type=type(error).__name__,error=str(error)[-2400:]);raise
    finally:
        if c.ssh_argv:
            try:
                for unit in c.UNITS:
                    s=c.props(unit)
                    if s['ActiveState']=='active' and s['SubState']!='exited':c.ssh('sudo systemctl stop '+unit,30)
                c.copyback(600)
            except Exception as error:c.record('copyback_error_disk_preserved',error=str(error)[-1500:])
        if guard and guard.poll() is None:
            (c.R/'stop-requested').write_text('closing\n')
            try:guard.wait(timeout=150)
            except subprocess.TimeoutExpired:c.record('guard_stop_observation_timeout',pid=guard.pid)
        if created:
            try:current=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone)
            except RuntimeError as error:
                if 'not found' not in str(error):raise
                current=None;c.record('no_vm_after_terminal_capacity_failure')
            if current:
                if vm:assert str(current['id'])==str(vm['id'])
                if current['status'] not in ('TERMINATED','STOPPED'):c.run(['gcloud','compute','instances','stop',c.NAME,'--zone='+zone,'--project='+c.PROJECT,'--quiet'],150)
                current=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);(c.R/'provider-final.json').write_text(json.dumps(current,indent=2)+'\n')
                elapsed=max(0,datetime.fromisoformat(current['lastStopTimestamp']).timestamp()-datetime.fromisoformat(current['lastStartTimestamp']).timestamp()) if current.get('lastStartTimestamp') and current.get('lastStopTimestamp') else 0
                c.record('provider_stopped',vm_id=str(current['id']),status=current['status'],focused_seconds_upper=2400+int((elapsed+99)//100*100),prior_gpu_seconds_upper=54451)
        awake.terminate()

if __name__=='__main__':main()
