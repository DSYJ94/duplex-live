"""One source-audit allocation; no optimizer path or duplicate submission."""
import json,os,shlex,subprocess,sys,time,hashlib,math
from pathlib import Path
from datetime import datetime
from . import controller as c
from .source_guard import AUTH,PLAN,KEY,sha
from portable.venus.guard import bounded_runpod_read


def prepare_ssh(vm,zone):
    for _ in range(30):
        try:keys=c.cloud('compute','instances','get-guest-attributes',c.NAME,'--zone='+zone,'--query-path=hostkeys/')
        except RuntimeError:keys=[]
        if keys:break
        time.sleep(3)
    else:raise RuntimeError('Google authenticated host keys unavailable')
    private=c.R/'.private';private.mkdir(mode=0o700);known=private/'known-hosts';alias='gcp-venus-source45'
    known.write_text(''.join(alias+' '+k['key']+' '+k['value']+'\n' for k in keys));known.chmod(0o600)
    proxy=shlex.join(['gcloud','compute','start-iap-tunnel',c.NAME,'22','--listen-on-stdin','--zone='+zone,'--project='+c.PROJECT,'--verbosity=error'])
    c.ssh_argv=['ssh','-i','runs/local-migration-20261024/.private/gcp-native-ssh','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','HostKeyAlias='+alias,'-o','UserKnownHostsFile='+str(known),'-o','ConnectTimeout=10','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2','-o','ControlMaster=auto','-o','ControlPersist=180','-o','ControlPath=/private/tmp/duplex-venus-source45-mux','-o','ProxyCommand='+proxy,'daryl_aurium_labs_com@'+vm['networkInterfaces'][0]['accessConfigs'][0]['natIP']]
    (c.R/'ssh.json').write_text(json.dumps(c.ssh_argv)+'\n')
    for _ in range(20):
        try:c.ssh('test -d '+c.ROOT+' && nvidia-smi',20);return
        except RuntimeError:time.sleep(3)
    raise RuntimeError('Strict SSH/GPU not ready')


def main():
    c.R=Path('runs/venus-training-20261101/source-audit-v45')
    if c.R.exists():raise ValueError('Observe the existing operation; never restart this allocation')
    p=json.loads(PLAN.read_text());price=json.loads(Path('runs/venus-training-20261101/current-a100-price.json').read_text())
    assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    assert not c.run(['git','-c','core.filemode=false','status','--porcelain','--','training/venus','data/venus','portable/venus','docs/NATIVE-VENUS-GCP-SOURCE-AUDIT-AUTHORIZATION-2026-11-01.json','docs/NATIVE-VENUS-GCP-SOURCE-AUDIT-EXECUTION-2026-11-01.json']).strip()
    zone='us-central1-a';c.NAME=p['vm_name'];duration=1200
    instances=c.cloud('compute','instances','list');assert all(v['status']=='TERMINATED' for v in instances)
    old=c.cloud('compute','instances','describe','duplex-venus-runtime-20261031-v44','--zone='+zone)
    assert old['id']==p['retire_only_vm_id'] and old['lastStopTimestamp']==p['previous_provider_stop'] and old['status']=='TERMINATED'
    disk=c.cloud('compute','disks','describe','duplex-venus-train-20261026-v1','--zone='+zone)
    assert disk['id']=='4375908109499109167' and int(disk['sizeGb'])==100
    q={v['metric']:v for v in c.cloud('compute','regions','describe','us-central1')['quotas']}
    assert q['NVIDIA_A100_GPUS']['usage']==0 and q['NVIDIA_A100_GPUS']['limit']>=1
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
    assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in pods)
    from data.venus.context_source_quality import inventory
    sources=Path('runs/venus-training-20261101/context-source-v1');payload=Path('docs/NATIVE-VENUS-CONTEXT-REPAIR-SPEECH-PAYLOAD-2026-11-01.json')
    rows=inventory(sources,payload);assert len(rows)==72 and sha(payload)==p['source_payload_sha256']
    backup=json.loads(Path('runs/venus-training-20261029/action-v44/copyback-verified.json').read_text());assert backup['all_hashes_verified']
    checkpoint=Path('runs/venus-training-20261029/action-v44/copyback/action-candidate-v42/step-0108.pt')
    assert sha(checkpoint)=='a934ce7e05f4728a0973408eafedaed219f89dacfbf5dc6fcb2bbf536f4378b6'
    c.R.mkdir(parents=True);c.source=c.run(['git','rev-parse','HEAD']).strip();c.remote=c.ROOT+'/duplex-source-'+c.source[:12]
    c.run(['git','bundle','create',str(c.R/'source.bundle'),'HEAD'],120)
    c.record('source_audit_admitted',source=c.source,prior_focused_seconds=28000,maximum_seconds=1200,optimizer_updates=0,source_count=72,hourly_usd=price['hourly_estimate_usd'])
    created=False;guard=None;awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())])
    remote_out=c.ROOT+'/venus-source-audit-v45';unit='duplex-venus-source45-words'
    try:
        c.cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=1h')
        c.run(['gcloud','compute','instances','delete',old['name'],'--zone='+zone,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
        assert not c.cloud('compute','disks','describe',disk['name'],'--zone='+zone).get('users')
        bootstrap=c.R/'bootstrap.sh';bootstrap.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-source45-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
        op=c.cloud('compute','instances','create',c.NAME,'--zone='+zone,'--machine-type=a2-highgpu-1g','--provisioning-model=FLEX_START','--request-valid-for-duration=900s','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+disk['name']+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(bootstrap),'--max-run-duration=1200s','--instance-termination-action=STOP','--async');op=op[0] if isinstance(op,list) else op;created=True
        c.record('one_source_allocation',operation=op['name'])
        until=time.time()+1000
        while op['status']!='DONE':
            if time.time()>until:raise TimeoutError('Observe SAME provider operation; never resubmit')
            time.sleep(5);op=c.cloud('compute','operations','describe',op['name'],'--zone='+zone)
        if op.get('error'):raise RuntimeError(json.dumps(op['error']))
        vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);assert vm['status']=='RUNNING'
        end=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()+duration
        (c.R/'provider-start.json').write_text(json.dumps(vm,indent=2)+'\n')
        prepare_ssh(vm,zone);c.sync(c.R/'source.bundle',c.ROOT+'/source45.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/source45.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+c.source)
        lease=dict(resource_key=KEY,zone=zone,vm_id=vm['id'],disk_id=disk['id'],disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=end,prior_new_gpu_seconds_upper=28000,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=sha(AUTH),execution_plan_sha256=sha(PLAN),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        lease_path=c.ROOT+'/source45-lease.json';(c.R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');c.sync(c.R/'lease.json',lease_path)
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(c.R/'lease.json'),'--ssh-argv',str(c.R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(c.R/'guard.json'),'--stop-file',str(c.R/'stop-requested')],stdout=(c.R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});c.guard=guard
        until=time.time()+90
        while not (c.R/'guard.json').exists():
            if guard.poll() is not None or time.time()>until:raise RuntimeError('Independent Mac guard not ready')
            time.sleep(1)
        live=json.loads(c.ssh('cat '+c.ROOT+'/venus-live-guard-v1.json'));assert live['vm_id']==vm['id'] and time.time()<live['expires_unix'] and live['stop_unix']==end
        c.ssh('sudo systemd-run --unit=duplex-source45-guest --property=RuntimeMaxSec=1300 --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lease_path)
        assert c.props('duplex-source45-guest')['ActiveState']=='active';c.ssh('sudo systemctl stop duplex-source45-bootstrap-stop.timer');c.record('both_guards_live',mac_pid=guard.pid)
        c.ssh('mkdir -p '+remote_out);c.sync(str(sources)+'/',remote_out+'/sources/')
        assert not c.ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits').strip()
        assert c.ssh('sha256sum '+c.OUT+'/action-candidate-v42/step-0108.pt').split()[0]==sha(checkpoint)
        py=c.ROOT+'/venus-runtime-v1/venv/bin/python'
        command='/usr/bin/env DUPLEX_LIVE_GUARD_FILE='+c.ROOT+'/venus-live-guard-v1.json DUPLEX_LIVE_LEASE_FILE='+lease_path+' HF_HOME='+c.ROOT+'/venus-runtime-v1/hf-cache '+py+' -u -m data.venus.context_source_quality --root '+remote_out+'/sources --payload '+c.remote+'/'+str(payload)+' --output '+remote_out+'/quality'
        assert end-time.time()>600,'Whole72 and copyback reserve required'
        c.start_unit(unit,command,int(end-time.time()-180));c.wait_unit(unit,end-180)
        report=json.loads(c.ssh('cat '+remote_out+'/quality/report.json'));assert report['required']==72 and report['word_audit_complete']
        c.record('all72_actual_words_terminal',all_pass=report['all72_pass'],training_approved=False)
    except BaseException as error:
        c.record('source_audit_failure',error_type=type(error).__name__,error=str(error)[-1500:]);raise
    finally:
        if c.ssh_argv:
            try:
                for u in c.UNITS:
                    state=c.props(u)
                    if state['ActiveState']=='active' and state['SubState']!='exited':c.ssh('sudo systemctl stop '+u)
                c.ssh('mkdir -p '+remote_out+'; sudo journalctl -u '+unit+' --no-pager -o cat > '+remote_out+'/journal.log')
                c.run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=core','--exclude=core.*','-e',shlex.join(c.ssh_argv[:-1]),c.ssh_argv[-1]+':'+remote_out+'/',str(c.R/'copyback')+'/'],180)
                script="import pathlib,hashlib,json; root=pathlib.Path("+repr(remote_out)+"); print(json.dumps([dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob('*')) if p.is_file()]))"
                items=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote(script)))
                assert items
                for item in items:
                    target=c.R/'copyback'/item['path'];assert target.stat().st_size==item['bytes'] and sha(target)==item['sha256']
                receipt=dict(all_hashes_verified=True,files=len(items),bytes=sum(x['bytes'] for x in items),inventory=items)
                (c.R/'copyback-verified.json').write_text(json.dumps(receipt,indent=2)+'\n')
                c.record('source_results_full_hash_verified',files=len(items),bytes=receipt['bytes'])
            except Exception as error:c.record('copyback_failed_disk_preserved',error_type=type(error).__name__)
        if created:
            try:
                vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone)
                if vm['status'] not in ('TERMINATED','STOPPED'):c.run(['gcloud','compute','instances','stop',c.NAME,'--zone='+zone,'--project='+c.PROJECT,'--quiet','--discard-local-ssd=True'],150)
                vm=c.cloud('compute','instances','describe',c.NAME,'--zone='+zone);(c.R/'provider-final.json').write_text(json.dumps(vm,indent=2)+'\n');assert vm['status']=='TERMINATED'
                elapsed=datetime.fromisoformat(vm['lastStopTimestamp']).timestamp()-datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();c.record('provider_stopped',actual_seconds=elapsed,focused_upper_seconds=28000+math.ceil(elapsed/100)*100)
            except Exception as error:c.record('provider_final_read_failed',error_type=type(error).__name__)
        if guard and guard.poll() is None:
            (c.R/'stop-requested').write_text('closed\n')
            try:guard.wait(timeout=35)
            except subprocess.TimeoutExpired:guard.terminate()
        awake.terminate()

if __name__=='__main__':main()
