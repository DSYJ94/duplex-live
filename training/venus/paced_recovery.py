"""Resume the stopped prepared run, preserving the SAME disk and all used time."""
import hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from . import controller as c
from .paced_guard import AUTH as ORIGINAL_AUTH,PLAN_V24,PLAN_V25,PLAN_V26,PLAN_V27,PLAN_V28,PLAN_V29,LIVE_AUTH
from portable.venus.guard import bounded_runpod_read

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    repaired_live='--repaired-live' in sys.argv
    AUTH=LIVE_AUTH if repaired_live else ORIGINAL_AUTH
    retry_live='--retry-live' in sys.argv
    live_trial='--live-trial' in sys.argv or retry_live or repaired_live
    evaluation_only='--evaluation-only' in sys.argv
    assert sum(flag in sys.argv for flag in ('--live-trial','--retry-live','--repaired-live','--evaluation-only','--prepared-reactive'))<=1
    prepared='--prepared-reactive' in sys.argv or evaluation_only or live_trial
    version='v29' if repaired_live else 'v28' if retry_live else 'v27' if live_trial else 'v26' if evaluation_only else 'v25' if prepared else 'v24'
    PLAN=PLAN_V29 if repaired_live else PLAN_V28 if retry_live else PLAN_V27 if live_trial else PLAN_V26 if evaluation_only else PLAN_V25 if prepared else PLAN_V24
    live_plan=json.loads(PLAN.read_text()) if live_trial else None
    prior,duration=(live_plan['focused_gpu_seconds_upper_before_launch'],live_plan['maximum_gpu_seconds']) if live_trial else (10500,3900) if evaluation_only else (9900,4500) if prepared else (6500,7200)
    unit_prefix='duplex-venus-recovery'+version[1:]
    c.R=Path('runs/venus-training-20261027')/('paced-recovery-'+version);assert not c.R.exists(),'Observe actual recovery, never duplicate'
    assert not c.run(['git','status','--porcelain','--','training/venus','data/venus','portable/venus']).strip()
    p=json.loads(PLAN.read_text());a=json.loads(AUTH.read_text());r=a['resources'];zone=r['zone'];c.NAME=p['vm_name']
    assert p['status']=='WITHIN_APPROVED_FOCUSED_REPAIR' and p['parent_sha256']==digest(AUTH)
    previous=Path('runs/venus-training-20261027')/('paced-recovery-v28' if repaired_live else 'paced-recovery-v27' if retry_live else 'paced-recovery-v26' if live_trial else 'paced-recovery-v25' if evaluation_only else 'paced-recovery-v24' if prepared else 'paced-continuation-v23')
    events=json.loads((previous/'events.json').read_text());last=json.loads((previous/'provider-final.json').read_text())
    backup_run=Path('runs/venus-training-20261027/paced-recovery-v26') if (retry_live or repaired_live) else previous
    if live_trial:
        import socket
        with socket.socket() as probe:probe.bind(('127.0.0.1',8037))
        final=next(e for e in reversed(events) if e['event']=='provider_stopped')
        assert final['focused_seconds_upper']==prior
        assert (prior,duration)==(13700,3600) if repaired_live else duration==min(900,14400-prior) and duration>=600
        assert json.loads((backup_run/'copyback-verified.json').read_text())['all_hashes_verified']
        assert digest(backup_run/'copyback/paced-candidate-v25/step-0054.pt')==p['trained_checkpoint_sha256']
        assert json.loads((backup_run/'copyback/paced-native-paired-v26/paired-native-report.json').read_text())['complete']
        assert json.loads((backup_run/'copyback/voice-source-quality-v26/quality.json').read_text())['complete']
    elif evaluation_only:
        assert any(e['event']=='actual_training_complete' and e['result']['actual_updates']==54 and e['result']['total_optimizer_updates']==244 for e in events)
        assert any(e['event']=='provider_stopped' and e['focused_seconds_upper']==10500 for e in events)
        assert digest(backup_run/'copyback/paced-candidate-v25/step-0054.pt')==p['trained_checkpoint_sha256']
        elapsed=datetime.fromisoformat(last['lastStopTimestamp']).timestamp()-datetime.fromisoformat(last['lastStartTimestamp']).timestamp()
        assert 9900+elapsed<=10500
    elif prepared:
        assert any(e['event']=='actual_unit_terminal' and e['unit']=='duplex-venus-recovery24-prefix' and e['state']['ExecMainStatus']=='0' for e in events)
        assert any(e['event']=='provider_stopped' and e['focused_seconds_upper']==9900 for e in events)
        candidate=previous/'copyback/paced-candidate-v24'
        assert not (candidate/'recipe.json').exists() and not list(candidate.glob('*.pt'))
        assert json.loads((candidate/'no-update-gradient-benchmark.json').read_text())['optimizer_updated'] is False
        elapsed=datetime.fromisoformat(last['lastStopTimestamp']).timestamp()-datetime.fromisoformat(last['lastStartTimestamp']).timestamp()
        assert 6500+elapsed<=9900
    else:
        assert any(e['event']=='actual_unit_terminal' and e['unit']=='duplex-venus-paced-native-cache-v23' and e['state']['ExecMainStatus']=='0' for e in events)
        assert not any('numerical-start' in e.get('unit','') or 'restore-train' in e.get('unit','') for e in events),'No old optimizer work may be repeated'
        elapsed=datetime.fromisoformat(last['lastStopTimestamp']).timestamp()-datetime.fromisoformat(last['lastStartTimestamp']).timestamp();assert 2600+elapsed<=6500
    for key in ('approval','manifest','source_audit','semantic_quality','prepared_semantic','word_quality'):assert digest(p[key])==p[key+'_sha256']
    vms=c.cloud('compute','instances','list');assert all(v['status']=='TERMINATED' for v in vms)
    old=next(v for v in vms if str(v['id'])==p['retire_only_vm_id']);assert old['name']==p['retire_only_vm_name'] and old['lastStartTimestamp']==last['lastStartTimestamp'] and old['lastStopTimestamp']==last['lastStopTimestamp']
    disk=c.cloud('compute','disks','describe',r['same_existing_disk_name'],'--zone='+zone);assert str(disk['id'])==r['same_existing_disk_id'] and int(disk['sizeGb'])==100
    assert any(d.get('boot') and not d.get('autoDelete') and d['source'].endswith('/disks/'+r['same_existing_disk_name']) for d in old['disks'])
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods;assert all(x['status'] in ('EXITED','STOPPED','TERMINATED') for x in pods)
    price=json.loads(Path('runs/venus-20261026/us-current-price40-v1.json').read_text());assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    seed=backup_run if prepared else Path('runs/venus-training-20261027/cadence-qualify-v21');proof=json.loads((seed/'copyback-verified.json').read_text());assert proof['all_hashes_verified']
    rows=json.loads((seed/'unique-inventory.json').read_text());assert len(rows)==p['backup_before_retirement']['files'] and sum(x['bytes'] for x in rows)==p['backup_before_retirement']['bytes']
    # Critical trained state is independently saved on this Mac before replacing only the stopped VM envelope.
    checkpoint=seed/'copyback/candidate-v13/step-0190.pt';assert digest(checkpoint)==p['parent_checkpoint_sha256']
    for row in rows:
        path=seed/'copyback'/row['path'];assert path.stat().st_size==row['bytes'] and digest(path)==row['sha256']
    c.R.mkdir();c.source=c.run(['git','rev-parse','HEAD']).strip();c.remote=c.ROOT+'/duplex-venus-recovery-'+c.source[:12];c.run(['git','bundle','create',str(c.R/'source.bundle'),'HEAD'],120)
    c.record('recovery_preflight',source=c.source,prior_focused_seconds=prior,next_max_seconds=duration,trained_state190_mac_hash_verified=True,no_optimizer_started_in_predecessor=not (evaluation_only or live_trial),evaluation_only=evaluation_only,live_trial=live_trial,same_disk=str(disk['id']))
    awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())]);guard=None;created=False;vm=None
    try:
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=4h')
        if retry_live:
            op=c.cloud('compute','instances','start',c.NAME,'--zone='+zone,'--async');op=op[0] if isinstance(op,list) else op;created=True;c.record('one_warm_live_retry',operation=op['name'],same_vm_id=str(old['id']))
        else:
            c.run(['gcloud','compute','instances','delete',old['name'],'--zone='+zone,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
            disk=c.cloud('compute','disks','describe',r['same_existing_disk_name'],'--zone='+zone);assert str(disk['id'])==r['same_existing_disk_id'] and not disk.get('users')
            c.record('only_stopped_envelope_retired_same_disk_contents_preserved',old_id=str(old['id']),disk_id=str(disk['id']))
            startup=c.R/'bootstrap.sh';startup.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit='+unit_prefix+'-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
            op=c.cloud('compute','instances','create',c.NAME,'--zone='+zone,'--machine-type=a2-highgpu-1g','--provisioning-model=FLEX_START','--request-valid-for-duration=900s','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+r['same_existing_disk_name']+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(startup),'--max-run-duration='+str(duration)+'s','--instance-termination-action=STOP','--async');op=op[0] if isinstance(op,list) else op;created=True;c.record('one_recovery_allocation',operation=op['name'],source=c.source)
        until=time.time()+960
        while op['status']!='DONE':
            if time.time()>until:raise TimeoutError('Observe same pending operation; never resubmit')
            time.sleep(5);op=c.cloud('compute','operations','describe',op['name'],'--zone='+zone)
        if op.get('error'):raise RuntimeError(json.dumps(op['error']))
        vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);assert vm['status']=='RUNNING' and ((str(vm['id'])==str(old['id'])) if retry_live else (str(vm['id'])!=str(old['id'])))
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+duration;assert int(vm['scheduling']['maxRunDuration']['seconds'])==duration
        c.record('actual_recovery_running',vm_id=str(vm['id']),same_disk_id=str(disk['id']),last_start=vm['lastStartTimestamp'],native_stop_unix=end)
        for _ in range(30):
            try:keys=c.cloud('compute','instances','get-guest-attributes',c.NAME,'--zone='+zone,'--query-path=hostkeys/')
            except RuntimeError:keys=[]
            if keys:break
            time.sleep(3)
        else:raise RuntimeError('Google hostkeys unavailable')
        private=c.R/'.private';private.mkdir(mode=0o700);known=private/'known-hosts';alias='gcp-venus-recovery-'+version;known.write_text(''.join(alias+' '+k['key']+' '+k['value']+'\n' for k in keys));known.chmod(0o600)
        c.ssh_argv=json.loads((previous/'ssh.json').read_text());c.ssh_argv=['HostKeyAlias='+alias if x.startswith('HostKeyAlias=') else 'UserKnownHostsFile='+str(known) if x.startswith('UserKnownHostsFile=') else ('ControlPath=/private/tmp/duplex-venus-'+version if x.startswith('ControlPath=') else x).replace('start-iap-tunnel '+old['name']+' ','start-iap-tunnel '+c.NAME+' ') for x in c.ssh_argv];c.ssh_argv[-1]=c.ssh_argv[-1].split('@',1)[0]+'@'+vm['networkInterfaces'][0]['accessConfigs'][0]['natIP'];(c.R/'ssh.json').write_text(json.dumps(c.ssh_argv)+'\n')
        for _ in range(20):
            try:c.ssh('test -f '+c.OUT+'/paced-native-cache-v22/manifest.json && nvidia-smi',20);break
            except RuntimeError:time.sleep(3)
        else:raise RuntimeError('Strict SSH not ready')
        bundle=c.ROOT+'/recovery-'+version+'-source.bundle';lease_path=c.ROOT+'/recovery-'+version+'-lease.json';c.sync(c.R/'source.bundle',bundle);c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+bundle+' HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+c.source)
        lease=dict(resource_key='venus-paced-recovery-'+version,zone=zone,vm_id=str(vm['id']),disk_id=str(disk['id']),disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=prior,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=digest(AUTH),execution_plan_sha256=digest(PLAN),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip());(c.R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');c.sync(c.R/'lease.json',lease_path)
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(c.R/'lease.json'),'--ssh-argv',str(c.R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(c.R/'guard.json'),'--stop-file',str(c.R/'stop-requested')],stdout=(c.R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});c.guard=guard
        until=time.time()+90
        while not (c.R/'guard.json').exists():
            if guard.poll() is not None:raise RuntimeError('Mac guard ended')
            if time.time()>until:raise TimeoutError('Guard readiness timeout')
            time.sleep(1)
        live=json.loads(c.ssh('cat '+c.ROOT+'/venus-live-guard-v1.json'));assert live['vm_id']==str(vm['id']) and time.time()<live['expires_unix'] and live['stop_unix']==end
        guest=unit_prefix+'-guest-guard';c.ssh('sudo systemd-run --unit='+guest+' --property=RuntimeMaxSec='+str(duration+100)+' --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lease_path);assert c.props(guest)['ActiveState']=='active';c.ssh('sudo systemctl stop '+('duplex-venus-recovery27' if retry_live else unit_prefix)+'-bootstrap-stop.timer');c.record('both_guards_live',mac_pid=guard.pid)
        dest=c.R/'copyback';dest.mkdir()
        for row in rows:
            original=seed/'copyback'/row['path'];target=dest/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(original,target)
            except OSError:shutil.copy2(original,target)
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python'
        cache=c.OUT+'/paced-prefix-cache-v24'
        if prepared:
            local_cache=seed/'copyback/paced-prefix-cache-v24/manifest.json'
            manifest=json.loads(local_cache.read_text())
            assert len(manifest['scenes'])==268 and not manifest['proof_only']
            assert manifest['prefix_producer']==p['prepared_cache_producer'] and manifest['source_manifest_sha256']==p['approved_timeline_manifest_sha256']
            assert all(x['native_prefix_proof']['passed'] for x in manifest['scenes'])
            assert c.ssh('sha256sum '+cache+'/manifest.json').split()[0]==digest(local_cache)==p['prepared_cache_sha256']
            assert c.ssh('sha256sum '+c.OUT+'/candidate-v13/step-0190.pt').split()[0]==p['parent_checkpoint_sha256']
            c.record('complete268_prefix_cache_reused',producer=manifest['prefix_producer'],proof='original successful terminal and exact copied manifest; each tensor SHA checked on read',reencoding=False,optimizer_updates=0,previous_backup_files=len(rows))
        else:
            for unit in ('duplex-venus-paced-native-cache-v23','duplex-venus-paced-prefix-cache-v23','duplex-venus-paced-guest-guard-v23'):
                c.ssh('mkdir -p '+c.OUT+'/journals; sudo journalctl -u '+unit+' --no-pager -o cat > '+c.OUT+'/journals/'+unit+'.log')
            native=json.loads(c.ssh('cat '+c.OUT+'/paced-native-cache-v22/manifest.json'));assert len(native['scenes'])==268 and not native['proof_only'] and native['source_manifest_sha256']==p['approved_timeline_manifest_sha256']
            assert c.ssh('sha256sum '+c.OUT+'/candidate-v13/step-0190.pt').split()[0]==p['parent_checkpoint_sha256']
            witness=c.OUT+'/paced-prefix-v23-reuse-witness.json';py=c.ROOT+'/venus-runtime-v1/venv/bin/python'
            c.ssh('cd '+c.remote+' && '+py+' -m training.venus.paced_reuse_witness --cache '+c.OUT+'/paced-native-cache-v22 --prefix '+c.OUT+'/paced-prefix-cache-v22 --journal '+c.OUT+'/journals/duplex-venus-paced-prefix-cache-v23.log --original-worktree '+c.ROOT+'/duplex-venus-paced-d6e3b9ffb20b --execution-plan '+c.remote+'/'+str(PLAN)+' --output '+witness,120)
            c.copyback(600);c.record('full_preoptimizer_backup_verified',native_cache_scenes=268,trained_state190_preserved=True)
        env='CUBLAS_WORKSPACE_CONFIG=:4096:8 DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+lease_path+' HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache'
        def execute(label,args,reserve):
            unit=unit_prefix+'-'+label;c.start_unit(unit,'/usr/bin/env '+env+' '+py+' -u -m '+args,max(1,int(end-time.time()-reserve)));c.wait_unit(unit,end-reserve)
        if live_trial:
            if retry_live:
                (c.R/'prior-microphone.log').write_text(c.ssh('sudo journalctl -u duplex-venus-recovery27-microphone --no-pager -o cat | tail -30'))
            from .live_trial import serve
            serve(c,'/usr/bin/env '+env+' '+py,end,unit_prefix)
        else:
            if not prepared:execute('prefix','training.venus.prefix_cache --model '+c.MODEL+' --cache '+c.OUT+'/paced-native-cache-v22 --output '+c.OUT+'/paced-prefix-cache-v24 --source '+c.source+' --reuse-cache '+c.OUT+'/paced-prefix-cache-v22 --reuse-witness '+witness,4000)
            if not evaluation_only:
                args=' --model '+c.MODEL+' --cache '+c.OUT+'/paced-prefix-cache-v24 --approval '+c.OUT+'/paced-data-approval-v22.json --output '+c.OUT+'/paced-candidate-'+version+' --source '+c.source+' --cache-producer '+(p['prepared_cache_producer'] if prepared else c.source)+' --lease '+lease_path+' --parent-checkpoint '+c.OUT+'/candidate-v13/step-0190.pt --parent-sha256 '+p['parent_checkpoint_sha256']
                if prepared:args+=' --reactive'
                execute('numerical-start','training.venus.train --stage start'+args,3600);recipe=json.loads(c.ssh('cat '+c.OUT+'/paced-candidate-'+version+'/recipe.json'));c.record('actual_recipe_admitted',recipe=recipe['recipe'],admission=recipe['admission'])
                execute('restore-train','training.venus.train --stage restore-continue'+args,3400);terminal=json.loads(c.ssh('cat '+c.OUT+'/paced-candidate-'+version+'/training-terminal.json'));assert terminal['completed'] and not terminal['adopted'];c.record('actual_training_complete',result=terminal)
            candidate_version='v25' if evaluation_only else version
            args='training.venus.paced_behavior --model '+c.MODEL+' --candidate '+c.OUT+'/paced-candidate-'+candidate_version+' --parent-checkpoint '+c.OUT+'/candidate-v13/step-0190.pt --parent-sha256 '+p['parent_checkpoint_sha256']+' --plan '+c.remote+'/evals/fixtures/venus-control-repair-source-v2-20261027.json --timelines '+c.OUT+'/paced-timelines-v22 --approval '+c.OUT+'/paced-data-approval-v22.json --word-quality '+c.OUT+'/paced-word-quality-v22.json --old-plan '+c.remote+'/evals/fixtures/venus-control-corpus-v4-20261026.json --old-timelines '+c.OUT+'/timelines-v7-repaired --old-approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/paced-native-paired-'+version+' --source '+c.source+' --lease '+lease_path
            execute('paired-native',args,1500);c.record('complete64_paired_native_collected',adopted=False)
            execute('voice-words','data.venus.voice_source_quality --root '+c.OUT+'/voice-source-export-v1 --output '+c.OUT+'/voice-source-quality-'+version+' --lease '+lease_path+' --source '+c.source,600)
            quality=json.loads(c.ssh('cat '+c.OUT+'/voice-source-quality-'+version+'/quality.json'));assert quality['complete'];c.record('complete174_source_words',passed=quality['passed'],required=174,all_passed=quality['all_passed'],adopted=False)
    except BaseException as error:c.record('recovery_failure',error_type=type(error).__name__,error=str(error)[-2400:]);raise
    finally:
        if c.ssh_argv:
            try:
                for unit in c.UNITS:
                    state=c.props(unit)
                    if state['ActiveState']=='active' and state['SubState']!='exited':c.ssh('sudo systemctl stop '+unit,30)
                c.copyback(600)
            except Exception as error:c.record('copyback_failure_disk_preserved',error=str(error)[-1500:])
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
                elapsed=max(0,datetime.fromisoformat(current['lastStopTimestamp']).timestamp()-datetime.fromisoformat(current['lastStartTimestamp']).timestamp()) if current.get('lastStartTimestamp') and current.get('lastStopTimestamp') else 0;c.record('provider_stopped',vm_id=str(current['id']),status=current['status'],focused_seconds_upper=prior+int((elapsed+99)//100*100),prior_gpu_seconds_upper=54451)
        awake.terminate()

if __name__=='__main__':main()
