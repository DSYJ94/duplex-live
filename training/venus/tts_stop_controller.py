"""One approved same-VM TTS diagnosis. No optimizer/API/resource growth path."""
import json,os,shlex,subprocess,sys,time,math
from pathlib import Path
from datetime import datetime
from . import controller as c
from .source_controller import prepare_ssh
from .tts_stop_guard import AUTH,PLAN,KEY,sha
from portable.venus.guard import bounded_runpod_read
R=PLAN.parent
NAME='duplex-venus-source-20261101-v46'
ZONE='us-central1-a'


def copyback(remote_out,source):
    root='/Users/daryl/Documents/duplex-native-recovery-20261102/'+R.name
    code="import pathlib,json,hashlib;r=pathlib.Path("+repr(remote_out)+");print(json.dumps([dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()]))"
    links=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote("import pathlib,json,os;r=pathlib.Path("+repr(remote_out)+");print(json.dumps([dict(path=str(p.relative_to(r)),target=os.readlink(p)) for p in r.rglob('*') if p.is_symlink()]))"),60))
    assert not links,'Diagnostic output must not contain links'
    rows=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote(code),60));(R/'remote-inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
    from .cpu_resume import batches
    direct=[('ControlMaster=no' if x.startswith('ControlMaster=') else 'ControlPath=none' if x.startswith('ControlPath=') else 'ControlPersist=no' if x.startswith('ControlPersist=') else x) for x in c.ssh_argv]
    assert rows,'No output inventory; retain temporary disk'
    for row in rows:
        path=Path(row['path']);assert not path.is_absolute() and '..' not in path.parts and not {'.private','credentials','owner-conversations','session-token','API keys.md'}&set(path.parts)
    Path(root).mkdir(parents=True,exist_ok=True)
    for n,batch in enumerate(batches(sorted(rows,key=lambda x:not x['path'].endswith(('.json','.jsonl'))))):
        filelist=R/('copy-%03d.files'%n);filelist.write_bytes(b''.join(x['path'].encode()+b'\0' for x in batch))
        c.run(['/opt/homebrew/bin/rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=API keys.md','--exclude=core','--exclude=core.*','--from0','--files-from='+str(filelist),'-e',shlex.join(direct[:-1]),direct[-1]+':'+remote_out+'/',root+'/'],90)
    for row in rows:
        path=Path(root)/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    result=dict(full_hash_inventory_verified=True,files=len(rows),bytes=sum(x['bytes'] for x in rows),destination=root,source=source,complete_report_present=any(Path(x['path']).name=='report.json' for x in rows),report_paths=[x['path'] for x in rows if Path(x['path']).name=='report.json'])
    (R/'copyback-verified.json').write_text(json.dumps(result,indent=2)+'\n');c.record('native_stop_copyback_verified',**result)
    return result


def resume_import_failure(root):
    """Resume only the observed terminal missing-import failure before storage.

    This never permits a capacity/model retry or an observation-timeout retry.
    Keep the original claim and terminal receipt and admit one explicit resume.
    """
    rows=json.loads((root/'events.json').read_text())
    assert len(rows)==1 and rows[0]['event']=='tts_stop_failed'
    assert rows[0]['error_type']=='NameError' and rows[0]['error']=="name 'Path' is not defined"
    assert {p.name for p in root.iterdir()}=={'claim.json','events.json','bootstrap.sh','source.bundle'}
    claim=json.loads((root/'claim.json').read_text())
    process=subprocess.run(['/bin/ps','-p',str(claim['pid']),'-o','pid='],capture_output=True,text=True)
    assert process.returncode==1 and not process.stdout.strip(),'Original controller must be actually gone'
    with (root/'import-failure-resume.json').open('x') as f:
        json.dump(dict(original_claim=claim,original_events_sha256=sha(root/'events.json'),original_bundle_sha256=sha(root/'source.bundle'),reason='Terminal pre-storage NameError; zero storage/capacity/model attempts',resumed_by_pid=os.getpid()),f)
    (root/'source.bundle').rename(root/'failed-import-source.bundle')
    c.events[:]=rows


def main(placement=False,completion=False,completion_window=False,resume_preallocation_import=False,completion_capacity=False,completion_same_vm=False,completion_same_vm_retry=False,completion_flex=False):
    global AUTH,PLAN,KEY,R,NAME,ZONE
    if completion_same_vm_retry:completion_same_vm=True
    if completion_flex:
        assert not(placement or completion_window or completion_capacity or completion_same_vm or resume_preallocation_import)
        completion=True;placement=True
        from . import tts_completion_flex_guard as g
        AUTH,PLAN,KEY=g.AUTH,g.PLAN,g.KEY;R=PLAN.parent
    elif completion_same_vm:
        assert not(placement or completion_window or completion_capacity or resume_preallocation_import)
        completion=True
        from . import tts_completion_same_vm_guard as g
        AUTH,PLAN,KEY=(g.RETRY_AUTH,g.RETRY_PLAN,g.KEY) if completion_same_vm_retry else (g.AUTH,g.PLAN,g.KEY);R=PLAN.parent
    elif completion_capacity:
        completion=True;placement=True
        from . import tts_completion_capacity_guard as g
        AUTH,PLAN,KEY=g.AUTH,g.PLAN,g.KEY;R=PLAN.parent
    elif completion_window:
        completion=True;placement=True
        from . import tts_completion_window_guard as g
        AUTH,PLAN,KEY=g.AUTH,g.PLAN,g.KEY;R=PLAN.parent
    elif completion:
        placement=True
        from . import tts_completion_guard as g
        AUTH,PLAN,KEY=g.AUTH,g.PLAN,g.KEY;R=PLAN.parent
    elif placement:
        from . import tts_placement_guard as g
        AUTH,PLAN,KEY=g.AUTH,g.PLAN,g.KEY;R=PLAN.parent
    version='62' if completion_flex else '61' if completion_same_vm_retry else '60' if completion_same_vm else '59' if completion_capacity else '58' if completion_window else '57' if completion else '56' if placement else '55'
    duration=1200 if completion else 1800;prior=33500 if completion else 33100;required=24 if completion else 12
    a=json.loads(AUTH.read_text());proposal=json.loads(Path(a['proposal_path']).read_text())
    assert a['status']=='APPROVED' and a['human_approval']==('Approve queued Flex-start within existing caps' if completion_flex else 'Approve existing-VM retry within existing caps' if completion_same_vm else 'Approve C/B fallback within existing caps' if completion_capacity else 'Approve refreshed 20-minute comparison within existing caps' if completion_window else 'Approve 20-minute completion validation within existing caps' if completion else 'Approve fallback within existing caps' if placement else 'Approve 30 minutes within existing caps') and sha(a['proposal_path'])==a['proposal_sha256']
    if completion_same_vm:
        assert sha(proposal['prior_diagnostic_authority'])==proposal['prior_diagnostic_authority_sha256']
        assert sha(proposal['complete_evaluation_inherited_from'])==proposal['complete_evaluation_sha256']
        prior_a=json.loads(Path(proposal['prior_diagnostic_authority']).read_text())
        assert sha(prior_a['proposal_path'])==prior_a['proposal_sha256']
        proposal=json.loads(Path(prior_a['proposal_path']).read_text())
    if completion_capacity:
        assert sha(proposal['previous_proposal'])==proposal['previous_proposal_sha256']
        proposal=json.loads(Path(proposal['previous_proposal']).read_text())
    if placement:
        prior_authority=Path('docs/NATIVE-VENUS-TTS-STOP-DIAGNOSTIC-AUTHORIZATION-2026-11-02.json');assert sha(prior_authority)==proposal['prior_diagnostic_authority_sha256']
        prior_a=json.loads(prior_authority.read_text());assert sha(prior_a['proposal_path'])==prior_a['proposal_sha256']
        proposal=json.loads(Path(prior_a['proposal_path']).read_text())
    c.R=R;R.mkdir(parents=True,exist_ok=True);c.NAME=NAME
    resume=bool((R/'claim.json').exists())
    if resume:
        if not(completion_window and resume_preallocation_import):raise ValueError('Observe existing controller; never repeat allocation')
        resume_import_failure(R)
    else:assert not resume_preallocation_import,'No terminal failed claim to resume'
    source=c.run(['git','rev-parse','HEAD']).strip();c.source=source;c.remote=c.ROOT+'/duplex-tts-stop'+version+'-'+source[:12]
    assert not c.run(['git','-c','core.filemode=false','status','--porcelain','--','training/venus','portable/venus',str(AUTH),proposal['source_selection_path']]).strip(),'Committed source required'
    price_path='runs/venus-training-20261104/capacity-recon/current-a100-flex-price.json' if completion_flex else 'runs/venus-training-20261101/current-a100-price.json'
    price=json.loads(Path(price_path).read_text());disk_price=price['pd_balanced_GiB_month'] if completion_flex else price['rates']['6AE1-525F-8B80']['price']
    assert disk_price<=.1 and 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    instances=c.cloud('compute','instances','list');assert all(v['status']=='TERMINATED' for v in instances)
    old=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);assert old['id']=='3873014408897233554' and old['status']=='TERMINATED'
    disk=c.cloud('compute','disks','describe','duplex-venus-train-20261026-v1','--zone='+ZONE);assert disk['id']=='4375908109499109167' and int(disk['sizeGb'])==100
    quota={v['metric']:v for v in c.cloud('compute','regions','describe','us-central1')['quotas']};assert quota['NVIDIA_A100_GPUS']['usage']==0 and quota['NVIDIA_A100_GPUS']['limit']>=1
    if completion_flex:
        assert quota['PREEMPTIBLE_NVIDIA_A100_GPUS']['usage']==0 and quota['PREEMPTIBLE_NVIDIA_A100_GPUS']['limit']>=1
        cpu=quota['PREEMPTIBLE_CPUS'] if quota['PREEMPTIBLE_CPUS']['limit'] else quota['CPUS']
        assert cpu['limit']-cpu['usage']>=12
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods;assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in pods)
    candidate=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/context-v52/candidate');assert sha(candidate/'step-0060.pt')==proposal['candidate_checkpoint_sha256']
    assert sum((candidate/n).stat().st_size for n in ('step-0060.pt','recipe.json'))<=200*1024**2
    claim=dict(pid=os.getpid(),source=source,prior_focused_gpu_seconds_upper=prior)
    if completion_same_vm:
        started=time.time();claim.update(prestart_started_unix=started,prestart_deadline_unix=started+2700,maximum_prestart_attempts=3)
    if completion_same_vm_retry:claim['execution_version']=61
    with (R/('resumed-controller-claim.json' if resume else 'claim.json')).open('x') as f:json.dump(claim,f)
    c.run(['git','bundle','create',str(R/'source.bundle'),'HEAD'],120)
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())]);guard=None;attempted=False;backed=False;snap=None;allocated=None;remote_out=c.OUT+'/tts-'+('completion' if completion else 'placement' if placement else 'stop')+'-v'+version
    try:
        bootstrap=R/'bootstrap.sh';bootstrap.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-tts'+version+'-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl='+('2h' if completion_flex else '1h'))
        if placement:
            if completion_flex:
                from .tts_flex_allocation import allocate
                vm,disk,ZONE,snap,storage_deadline=allocate()
            else:
                from .tts_placement import allocate
                vm,disk,ZONE,snap,storage_deadline=allocate(version)
            NAME=vm['name'];c.NAME=NAME;attempted=True;allocated=(vm,disk,ZONE,snap,version)
        else:
            c.cloud('compute','instances','add-metadata',NAME,'--zone='+ZONE,'--metadata-from-file=startup-script='+str(bootstrap))
            c.cloud('compute','instances','set-scheduling',NAME,'--zone='+ZONE,'--max-run-duration='+str(duration)+'s','--instance-termination-action=STOP')
            attempted=True
            if completion_same_vm:
                from .tts_same_vm_start import start
                vm,actual_attempt=start(old,claim)
            else:
                op=c.cloud('compute','instances','start',NAME,'--zone='+ZONE,'--async');op=op[0] if isinstance(op,list) else op;c.record('one_tts_stop_gpu_attempt',operation=op['name'],source=source)
                until=time.time()+900
                while op['status']!='DONE':
                    assert time.time()<until,'Observe SAME operation; no resubmission'
                    time.sleep(4);op=c.cloud('compute','operations','describe',op['name'],'--zone='+ZONE)
                (R/'start-operation.json').write_text(json.dumps(op,indent=2)+'\n')
                if op.get('error'):raise RuntimeError(json.dumps(op['error']))
                vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);assert vm['status']=='RUNNING'
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+duration;(R/'provider-start.json').write_text(json.dumps(vm,indent=2)+'\n')
        prepare_ssh(vm,ZONE,version);c.sync(R/'source.bundle',c.ROOT+'/tts'+version+'.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/tts'+version+'.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+source)
        execution=dict(resource_key=KEY,authorization_sha256=sha(AUTH),new_optimizer_updates_authorized_by_this_execution=0,required_native_cases=required,maximum_gpu_seconds=duration,maximum_new_running_intervals=1)
        if placement:execution.update(vm_id=vm['id'],disk_id=disk['id'],snapshot_id=snap['id'],snapshot_source_disk_id=snap['sourceDiskId'],vm_name=NAME,disk_name=disk['name'])
        if completion_window or completion_capacity or completion_flex:execution.update(storage_started_unix=storage_deadline-7200,storage_deadline_unix=storage_deadline)
        if completion_flex:execution['maximum_queue_seconds']=3600
        if completion_same_vm:execution.update(prestart_started_unix=claim['prestart_started_unix'],prestart_deadline_unix=claim['prestart_deadline_unix'],maximum_prestart_attempts=3,actual_start_attempt_index=actual_attempt)
        if completion_same_vm_retry:execution['execution_version']=61
        PLAN.write_text(json.dumps(execution,indent=2)+'\n');c.ssh('mkdir -p '+c.remote+'/'+str(R));c.sync(PLAN,c.remote+'/'+str(PLAN))
        lease=dict(resource_key=KEY,zone=ZONE,vm_id=vm['id'],disk_id=disk['id'],disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=prior,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=sha(AUTH),execution_plan_sha256=sha(PLAN),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        if placement:lease.update(snapshot_id=snap['id'],storage_deadline_unix=storage_deadline)
        if completion_window or completion_capacity or completion_flex:lease.update(storage_started_unix=storage_deadline-7200)
        if completion_same_vm:lease.update(prestart_started_unix=claim['prestart_started_unix'],prestart_deadline_unix=claim['prestart_deadline_unix'])
        if completion_same_vm_retry:lease['execution_version']=61
        (R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');lp=c.ROOT+'/tts'+version+'-lease.json';c.sync(R/'lease.json',lp)
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(R/'lease.json'),'--ssh-argv',str(R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(R/'guard.json'),'--stop-file',str(R/'stop-requested')],stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT);c.guard=guard
        until=time.time()+90
        while not (R/'guard.json').exists():
            assert guard.poll() is None and time.time()<until;time.sleep(1)
        live=json.loads(c.ssh('cat '+c.ROOT+'/venus-live-guard-v1.json'));assert time.time()<live['expires_unix'] and live['vm_id']==vm['id']
        c.ssh('sudo systemd-run --unit=duplex-tts'+version+'-guest --property=RuntimeMaxSec='+str(duration+100)+' --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lp)
        assert c.props('duplex-tts'+version+'-guest')['ActiveState']=='active';c.ssh('sudo systemctl stop duplex-tts'+version+'-bootstrap-stop.timer');c.record('tts_stop_both_guards_live',mac_pid=guard.pid,native_stop_unix=end)
        c.ssh('mkdir -p '+c.ROOT+'/tts'+version+'-input')
        for name in ('step-0060.pt','recipe.json'):c.sync(candidate/name,c.ROOT+'/tts'+version+'-input/'+name)
        assert c.ssh('sha256sum '+c.ROOT+'/tts'+version+'-input/step-0060.pt').split()[0]==proposal['candidate_checkpoint_sha256']
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python'
        command='/usr/bin/env DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+lp+' HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache '+py+' -u -m training.venus.tts_stop_probe --model '+c.MODEL+' --checkpoint '+c.ROOT+'/tts'+version+'-input/step-0060.pt --selection '+c.remote+'/'+proposal['source_selection_path']+' --timelines '+c.OUT+'/paced-timelines-v22 --voice-reference '+c.OUT+'/george-reference-v33/reference-16k.wav --voice-reference-sha256 '+proposal['voice_reference_sha256']+' --lease '+lp+' --source '+source+' --output '+remote_out
        assert not c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').strip()
        if completion:
            # Both complete cohorts on the SAME new interval; never reuse an
            # old profiler run as the current counterfactual baseline.
            for label,flag in [('baseline',''),('completion',' --speech-completion')]:
                assert end-time.time()>600,'Full next12cases and copyback reserve required'
                unit='duplex-venus-tts-'+label+version
                c.start_unit(unit,command.replace('--output '+remote_out,'--output '+remote_out+'/'+label)+' --minimum-start-seconds=600'+flag,min(600,int(end-time.time()-120)))
                c.wait_unit(unit,end-120)
                report=json.loads(c.ssh('cat '+remote_out+'/'+label+'/report.json'))
                assert report['complete'] and len(report['rows'])==12 and report['new_optimizer_updates']==0
                if label=='completion':assert report['speech_completion'] and report['all_opt_in_tails_complete']
            c.record('complete24_native_completion_comparison',cases=24,adopted=False)
        else:
            assert end-time.time()>1000
            c.start_unit('duplex-venus-tts-stop'+version,command,min(1200,int(end-time.time()-180)));c.wait_unit('duplex-venus-tts-stop'+version,end-180)
            report=json.loads(c.ssh('cat '+remote_out+'/report.json'));assert report['complete'] and len(report['rows'])==12 and report['new_optimizer_updates']==0
            c.record('complete12_native_tts_stop',cases=12,adopted=False)
    except BaseException as error:
        c.record('tts_stop_failed',error_type=type(error).__name__,error=str(error)[-1800:]);raise
    finally:
        if c.ssh_argv:
            try:
                for unit in c.UNITS:
                    if c.props(unit).get('SubState')=='running':c.ssh('sudo systemctl stop '+unit,25)
                c.ssh('mkdir -p '+remote_out+'; '+''.join('sudo journalctl -u '+u+' --no-pager -o cat > '+remote_out+'/'+u+'.log; ' for u in c.UNITS))
                copyback(remote_out,source);backed=True
            except Exception as error:c.record('tts_stop_copyback_failed_preserve_original_disk',error_type=type(error).__name__,error=str(error)[-800:])
        if attempted:
            vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
            if vm['status']!='TERMINATED':c.run(['gcloud','compute','instances','stop',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet','--discard-local-ssd=True'],150)
            final=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);(R/'provider-final.json').write_text(json.dumps(final,indent=2)+'\n')
            actual=0 if final['lastStartTimestamp']==old['lastStartTimestamp'] else datetime.fromisoformat(final['lastStopTimestamp']).timestamp()-datetime.fromisoformat(final['lastStartTimestamp']).timestamp()
            c.record('tts_stop_gpu_terminal',actual_seconds=actual,upper_seconds=math.ceil(actual/100)*100,prior_focused_gpu_seconds_upper=prior,original_disk_preserved=True)
        if allocated and backed:
            from .tts_placement import retire_snapshot
            if completion_flex:
                from .tts_flex_allocation import retire
            else:
                from .tts_placement import retire
            retire(*allocated);retire_snapshot(snap)
        elif allocated:c.record('temporary_disk_preserved_unverified_results',disk_id=disk['id'],zone=ZONE,storage_deadline_unix=storage_deadline)
        if guard:
            try:guard.wait(timeout=25)
            except subprocess.TimeoutExpired:guard.terminate()
        awake.terminate()


if __name__=='__main__':main(placement='--placement' in sys.argv,completion='--completion' in sys.argv,completion_window='--completion-window' in sys.argv,resume_preallocation_import='--resume-preallocation-import' in sys.argv,completion_capacity='--completion-capacity' in sys.argv,completion_same_vm='--completion-same-vm' in sys.argv,completion_same_vm_retry='--completion-same-vm-retry' in sys.argv,completion_flex='--completion-flex' in sys.argv)
