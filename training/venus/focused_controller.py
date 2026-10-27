"""One short same-VM audit, with STOP and backup before any further work."""
import hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from . import controller as c
from .focused_guard import AUTH,NAME,IDENT,VOICE_PLAN,VOICE_PLAN_V17,VOICE_PLAN_V18
from portable.context.gcp_budget_watchdog import read_runpod_pods

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'audit-v15'
    assert mode in ('audit-v15','voice-v16','voice-v17','voice-v18')
    voice=mode.startswith('voice-');version=mode.rsplit('-',1)[-1]
    stage='voice-reference-'+version if voice else 'focused-audit-v15'
    key='venus-'+stage if voice else 'venus-focused-audit-v15'
    voice_plan={'v15':VOICE_PLAN,'v16':VOICE_PLAN,'v17':VOICE_PLAN_V17,'v18':VOICE_PLAN_V18}[version]
    c.R=Path('runs/venus-training-20261027')/stage;assert not c.R.exists(),'Observe existing run; never duplicate'
    assert not c.run(['git','status','--porcelain','--','training/venus','portable/venus/guard.py']).strip()
    a=json.loads(AUTH.read_text());r=a['resources'];c.NAME=NAME;zone=r['zone']
    vms=c.cloud('compute','instances','list');assert all(v['status']=='TERMINATED' for v in vms)
    vm=next(v for v in vms if str(v['id'])==IDENT);assert vm['name']==NAME and int(vm['scheduling']['maxRunDuration']['seconds'])==1300
    disk=c.cloud('compute','disks','describe',r['same_existing_disk_name'],'--zone='+zone);assert str(disk['id'])==r['same_existing_disk_id'] and int(disk['sizeGb'])==100
    pods=read_runpod_pods(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
    assert all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    price=json.loads(Path('runs/venus-20261026/us-current-price40-v1.json').read_text());assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    old=Path('runs/venus-training-20261027/behavior-v14');assert json.loads((old/'copyback-verified.json').read_text())['all_hashes_verified']
    prior={'v15':0,'v16':200,'v17':300,'v18':500}[version]
    if voice:
        preceding=Path('runs/venus-training-20261027')/{'v16':'focused-audit-v15','v17':'voice-reference-v16','v18':'voice-reference-v17'}[version];last=json.loads((preceding/'provider-final.json').read_text())
        assert vm['lastStartTimestamp']==last['lastStartTimestamp'] and vm['lastStopTimestamp']==last['lastStopTimestamp']
        elapsed=datetime.fromisoformat(last['lastStopTimestamp']).timestamp()-datetime.fromisoformat(last['lastStartTimestamp']).timestamp()
        assert 0<elapsed<=prior
    c.R.mkdir();c.source=c.run(['git','rev-parse','HEAD']).strip();c.remote=c.ROOT+'/duplex-venus-focused-'+c.source[:12]
    c.run(['git','bundle','create',str(c.R/'source.bundle'),'HEAD'],120)
    c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=2h')
    awake=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())]);guard=None;submitted=False
    try:
        op=c.cloud('compute','instances','start',NAME,'--zone='+zone,'--async');op=op[0] if isinstance(op,list) else op;submitted=True
        c.record('one_warm_audit_start',operation=op['name'],vm_id=IDENT,source=c.source,prior_gpu_seconds_upper=54451,new_scope_prior_seconds=prior)
        until=time.time()+900
        while op['status']!='DONE':
            if time.time()>until:raise TimeoutError('Observe same operation; never resubmit')
            time.sleep(3);op=c.cloud('compute','operations','describe',op['name'],'--zone='+zone)
        if op.get('error'):raise RuntimeError(json.dumps(op['error']))
        vm=c.cloud('compute','instances','describe',NAME,'--zone='+zone);assert vm['status']=='RUNNING' and str(vm['id'])==IDENT
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+1300
        c.record('actual_same_vm_audit_running',vm_id=IDENT,last_start=vm['lastStartTimestamp'],native_stop_unix=end)
        c.ssh_argv=json.loads((old/'ssh.json').read_text());c.ssh_argv[-1]='daryl_aurium_labs_com@'+vm['networkInterfaces'][0]['accessConfigs'][0]['natIP']
        if voice:c.ssh_argv=[x.replace('ControlPath=/private/tmp/duplex-venus-behavior-v14','ControlPath=/private/tmp/duplex-venus-voice-'+version) for x in c.ssh_argv]
        # Same disk/VM, previously Google-verified host keys; never relax checking.
        (c.R/'ssh.json').write_text(json.dumps(c.ssh_argv)+'\n')
        for _ in range(30):
            try:c.ssh('test -d '+c.OUT+'/native-holdout36-v14 && nvidia-smi',20);break
            except RuntimeError:time.sleep(3)
        else:raise RuntimeError('Strict SSH not ready')
        bundle=c.ROOT+'/'+stage+'-source.bundle';lease_path=c.ROOT+'/'+stage+'-lease.json';guest_unit='duplex-venus-'+stage+'-guest-guard'
        c.sync(c.R/'source.bundle',bundle)
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+bundle+' HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+c.source)
        lease=dict(resource_key=key,zone=zone,vm_id=IDENT,disk_id=str(disk['id']),disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=prior,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=hashlib.sha256(AUTH.read_bytes()).hexdigest(),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        if voice:lease['execution_plan_sha256']=hashlib.sha256(voice_plan.read_bytes()).hexdigest()
        (c.R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');c.sync(c.R/'lease.json',lease_path)
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(c.R/'lease.json'),'--ssh-argv',str(c.R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(c.R/'guard.json'),'--stop-file',str(c.R/'stop-requested')],stdout=(c.R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});c.guard=guard
        until=time.time()+90
        while not (c.R/'guard.json').exists():
            if guard.poll() is not None:raise RuntimeError('Independent Mac guard ended')
            if time.time()>until:raise TimeoutError('No verified guard')
            time.sleep(1)
        remote_guard=json.loads(c.ssh('cat '+c.ROOT+'/venus-live-guard-v1.json'))
        assert remote_guard['authorization_sha256']==lease['authorization_sha256'] and remote_guard['vm_id']==IDENT and remote_guard['stop_unix']==end and time.time()<remote_guard['expires_unix']
        c.ssh('sudo systemd-run --unit='+guest_unit+' --property=RuntimeMaxSec=1400 --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lease_path)
        assert c.props(guest_unit)['ActiveState']=='active'
        c.ssh('sudo systemctl stop duplex-venus-behavior-bootstrap-stop.timer');c.record('both_guards_live',mac_pid=guard.pid)
        dest=c.R/'copyback';dest.mkdir()
        seed=Path('runs/venus-training-20261027/focused-audit-v15') if voice else old
        for row in json.loads((seed/'unique-inventory.json').read_text()):
            original=seed/'copyback'/row['path']
            if not original.is_file():continue
            target=dest/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(original,target)
            except OSError:shutil.copy2(original,target)
        env='DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+lease_path+' HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache'
        if voice:
            c.copyback() # Retrieve actual missing audit files first; no recomputation.
            source_dir=Path(json.loads(voice_plan.read_text())['local_artifact_root'])
            c.sync(str(source_dir)+'/',c.OUT+'/voice-sources-v1/')
            cp=a['current_evidence']['checkpoint_sha256']
            args=' --execution-plan '+c.remote+'/'+str(voice_plan)+' --reference '+c.OUT+'/voice-sources-v1/legacy-eleven-reference.wav --model '+c.MODEL+' --checkpoint '+c.OUT+'/candidate-v13/step-0190.pt --checkpoint-sha256 '+cp+' --plan '+c.remote+'/evals/fixtures/venus-control-corpus-v4-20261026.json --timelines '+c.OUT+'/timelines-v7-repaired --approval '+c.OUT+'/data-approval-v7.json --output '+c.OUT+'/voice-comparison-'+version+' --source '+c.source+' --lease '+lease_path
            module='training.venus.voice_compare';unit='duplex-venus-voice-comparison-'+version
        else:
            module='training.venus.spoken_audit';args=' --collection '+c.OUT+'/native-holdout36-v14 --output '+c.OUT+'/spoken-audit36-v15 --source '+c.source+' --lease '+lease_path;unit='duplex-venus-spoken-audit36-v15'
        cmd='/usr/bin/env '+env+' '+c.ROOT+'/venus-runtime-v1/venv/bin/python -u -m '+module+args
        c.start_unit(unit,cmd,max(1,int(end-time.time()-210)));c.wait_unit(unit,end-210)
        filename='/voice-comparison-'+version+'/voice-comparison.json' if voice else '/spoken-audit36-v15/spoken-audit.json'
        result=json.loads(c.ssh('cat '+c.OUT+filename));assert result['complete'] and len(result['cases'])==(24 if voice else 36)
        c.record('complete_native_voice_comparison' if voice else 'complete36_spoken_audit',cases=len(result['cases']),qualified=False)
    except BaseException as e:c.record('audit_failure',error_type=type(e).__name__,error=str(e)[-2000:]);raise
    finally:
        if c.ssh_argv:
            try:
                for unit in c.UNITS:
                    s=c.props(unit)
                    if s['ActiveState']=='active' and s['SubState']!='exited':c.ssh('sudo systemctl stop '+unit,30)
                c.copyback()
            except Exception as e:c.record('copyback_failure_disk_preserved',error=str(e)[-1500:])
        if guard and guard.poll() is None:
            (c.R/'stop-requested').write_text('closing\n')
            try:guard.wait(timeout=150)
            except subprocess.TimeoutExpired:c.record('guard_stop_observation_timeout',pid=guard.pid)
        if submitted:
            current=c.cloud('compute','instances','describe',NAME,'--zone='+zone);assert str(current['id'])==IDENT
            if current['status'] not in ('TERMINATED','STOPPED'):c.run(['gcloud','compute','instances','stop',NAME,'--zone='+zone,'--project='+c.PROJECT,'--quiet'],150)
            current=c.cloud('compute','instances','describe',NAME,'--zone='+zone);(c.R/'provider-final.json').write_text(json.dumps(current,indent=2)+'\n')
            elapsed=max(0,datetime.fromisoformat(current['lastStopTimestamp']).timestamp()-datetime.fromisoformat(current['lastStartTimestamp']).timestamp())
            c.record('provider_stopped',vm_id=IDENT,status=current['status'],new_scope_actual_gpu_seconds=elapsed,new_scope_gpu_seconds_upper=prior+int((elapsed+99)//100*100),prior_gpu_seconds_upper=54451)
        awake.terminate()

if __name__=='__main__':main()
