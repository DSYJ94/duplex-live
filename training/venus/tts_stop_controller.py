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
    root='/Users/daryl/Documents/duplex-native-recovery-20261102/tts-stop-v55'
    code="import pathlib,json,hashlib;r=pathlib.Path("+repr(remote_out)+");print(json.dumps([dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()]))"
    rows=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote(code),60));(R/'remote-inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
    from .cpu_resume import batches
    direct=[('ControlMaster=no' if x.startswith('ControlMaster=') else 'ControlPath=none' if x.startswith('ControlPath=') else 'ControlPersist=no' if x.startswith('ControlPersist=') else x) for x in c.ssh_argv]
    Path(root).mkdir(parents=True,exist_ok=True)
    for n,batch in enumerate(batches(sorted(rows,key=lambda x:not x['path'].endswith(('.json','.jsonl'))))):
        filelist=R/('copy-%03d.files'%n);filelist.write_bytes(b''.join(x['path'].encode()+b'\0' for x in batch))
        c.run(['/opt/homebrew/bin/rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=API keys.md','--exclude=core','--exclude=core.*','--from0','--files-from='+str(filelist),'-e',shlex.join(direct[:-1]),direct[-1]+':'+remote_out+'/',root+'/'],90)
    for row in rows:
        path=Path(root)/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    result=dict(full_hash_inventory_verified=True,files=len(rows),bytes=sum(x['bytes'] for x in rows),destination=root,source=source,complete_report_present=any(x['path']=='report.json' for x in rows))
    (R/'copyback-verified.json').write_text(json.dumps(result,indent=2)+'\n');c.record('native_stop_copyback_verified',**result)


def main():
    a=json.loads(AUTH.read_text());proposal=json.loads(Path(a['proposal_path']).read_text())
    assert a['status']=='APPROVED' and a['human_approval']=='Approve 30 minutes within existing caps' and sha(a['proposal_path'])==a['proposal_sha256']
    c.R=R;R.mkdir(parents=True,exist_ok=True);c.NAME=NAME
    if (R/'claim.json').exists():raise ValueError('Observe existing controller; never repeat allocation')
    source=c.run(['git','rev-parse','HEAD']).strip();c.source=source;c.remote=c.ROOT+'/duplex-tts-stop55-'+source[:12]
    assert not c.run(['git','-c','core.filemode=false','status','--porcelain','--','training/venus','portable/venus',str(AUTH),proposal['source_selection_path']]).strip(),'Committed source required'
    price=json.loads(Path('runs/venus-training-20261101/current-a100-price.json').read_text());assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    instances=c.cloud('compute','instances','list');assert all(v['status']=='TERMINATED' for v in instances)
    old=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);assert old['id']=='3873014408897233554' and old['status']=='TERMINATED'
    disk=c.cloud('compute','disks','describe','duplex-venus-train-20261026-v1','--zone='+ZONE);assert disk['id']=='4375908109499109167' and int(disk['sizeGb'])==100
    quota={v['metric']:v for v in c.cloud('compute','regions','describe','us-central1')['quotas']};assert quota['NVIDIA_A100_GPUS']['usage']==0 and quota['NVIDIA_A100_GPUS']['limit']>=1
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods;assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in pods)
    candidate=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/context-v52/candidate');assert sha(candidate/'step-0060.pt')==proposal['candidate_checkpoint_sha256']
    assert sum((candidate/n).stat().st_size for n in ('step-0060.pt','recipe.json'))<=200*1024**2
    with (R/'claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),source=source,prior_focused_gpu_seconds_upper=33100),f)
    c.run(['git','bundle','create',str(R/'source.bundle'),'HEAD'],120)
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())]);guard=None;attempted=False;remote_out=c.OUT+'/tts-stop-v55'
    try:
        bootstrap=R/'bootstrap.sh';bootstrap.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-tts55-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=1h')
        c.cloud('compute','instances','add-metadata',NAME,'--zone='+ZONE,'--metadata-from-file=startup-script='+str(bootstrap))
        c.cloud('compute','instances','set-scheduling',NAME,'--zone='+ZONE,'--max-run-duration=1800s','--instance-termination-action=STOP')
        op=c.cloud('compute','instances','start',NAME,'--zone='+ZONE,'--async');op=op[0] if isinstance(op,list) else op;attempted=True;c.record('one_tts_stop_gpu_attempt',operation=op['name'],source=source)
        # Same operation only; no capacity or model retries.
        until=time.time()+900
        while op['status']!='DONE':
            assert time.time()<until,'Observe SAME operation; no resubmission'
            time.sleep(4);op=c.cloud('compute','operations','describe',op['name'],'--zone='+ZONE)
        (R/'start-operation.json').write_text(json.dumps(op,indent=2)+'\n')
        if op.get('error'):raise RuntimeError(json.dumps(op['error']))
        vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);assert vm['status']=='RUNNING'
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+1800;(R/'provider-start.json').write_text(json.dumps(vm,indent=2)+'\n')
        prepare_ssh(vm,ZONE,'55');c.sync(R/'source.bundle',c.ROOT+'/tts55.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/tts55.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+source)
        execution=dict(resource_key=KEY,authorization_sha256=sha(AUTH),new_optimizer_updates_authorized_by_this_execution=0,required_native_cases=12,maximum_gpu_seconds=1800,maximum_new_running_intervals=1)
        PLAN.write_text(json.dumps(execution,indent=2)+'\n');c.ssh('mkdir -p '+c.remote+'/'+str(R));c.sync(PLAN,c.remote+'/'+str(PLAN))
        lease=dict(resource_key=KEY,zone=ZONE,vm_id=vm['id'],disk_id=disk['id'],disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=33100,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=sha(AUTH),execution_plan_sha256=sha(PLAN),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        (R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');lp=c.ROOT+'/tts55-lease.json';c.sync(R/'lease.json',lp)
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(R/'lease.json'),'--ssh-argv',str(R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(R/'guard.json'),'--stop-file',str(R/'stop-requested')],stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT);c.guard=guard
        until=time.time()+90
        while not (R/'guard.json').exists():
            assert guard.poll() is None and time.time()<until;time.sleep(1)
        live=json.loads(c.ssh('cat '+c.ROOT+'/venus-live-guard-v1.json'));assert time.time()<live['expires_unix'] and live['vm_id']==vm['id']
        c.ssh('sudo systemd-run --unit=duplex-tts55-guest --property=RuntimeMaxSec=1900 --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lp)
        assert c.props('duplex-tts55-guest')['ActiveState']=='active';c.ssh('sudo systemctl stop duplex-tts55-bootstrap-stop.timer');c.record('tts_stop_both_guards_live',mac_pid=guard.pid,native_stop_unix=end)
        c.ssh('mkdir -p '+c.ROOT+'/tts55-input')
        for name in ('step-0060.pt','recipe.json'):c.sync(candidate/name,c.ROOT+'/tts55-input/'+name)
        assert c.ssh('sha256sum '+c.ROOT+'/tts55-input/step-0060.pt').split()[0]==proposal['candidate_checkpoint_sha256']
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python'
        command='/usr/bin/env DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+lp+' HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache '+py+' -u -m training.venus.tts_stop_probe --model '+c.MODEL+' --checkpoint '+c.ROOT+'/tts55-input/step-0060.pt --selection '+c.remote+'/'+proposal['source_selection_path']+' --timelines '+c.OUT+'/paced-timelines-v22 --voice-reference '+c.OUT+'/george-reference-v33/reference-16k.wav --voice-reference-sha256 '+proposal['voice_reference_sha256']+' --lease '+lp+' --source '+source+' --output '+remote_out
        assert not c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').strip()
        assert end-time.time()>1000
        c.start_unit('duplex-venus-tts-stop55',command,min(1200,int(end-time.time()-180)));c.wait_unit('duplex-venus-tts-stop55',end-180)
        report=json.loads(c.ssh('cat '+remote_out+'/report.json'));assert report['complete'] and len(report['rows'])==12 and report['new_optimizer_updates']==0
        c.record('complete12_native_tts_stop',cases=12,adopted=False)
    except BaseException as error:
        c.record('tts_stop_failed',error_type=type(error).__name__,error=str(error)[-1800:]);raise
    finally:
        if c.ssh_argv:
            try:
                for unit in c.UNITS:
                    if c.props(unit).get('SubState')=='running':c.ssh('sudo systemctl stop '+unit,25)
                copyback(remote_out,source)
            except Exception as error:c.record('tts_stop_copyback_failed_preserve_original_disk',error_type=type(error).__name__,error=str(error)[-800:])
        if attempted:
            vm=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE)
            if vm['status']!='TERMINATED':c.run(['gcloud','compute','instances','stop',NAME,'--zone='+ZONE,'--project='+c.PROJECT,'--quiet','--discard-local-ssd=True'],150)
            final=c.cloud('compute','instances','describe',NAME,'--zone='+ZONE);(R/'provider-final.json').write_text(json.dumps(final,indent=2)+'\n')
            actual=0 if final['lastStartTimestamp']==old['lastStartTimestamp'] else datetime.fromisoformat(final['lastStopTimestamp']).timestamp()-datetime.fromisoformat(final['lastStartTimestamp']).timestamp()
            c.record('tts_stop_gpu_terminal',actual_seconds=actual,upper_seconds=math.ceil(actual/100)*100,prior_focused_gpu_seconds_upper=33100,original_disk_preserved=True)
        if guard:
            try:guard.wait(timeout=25)
            except subprocess.TimeoutExpired:guard.terminate()
        awake.terminate()


if __name__=='__main__':main()
