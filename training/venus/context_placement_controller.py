"""Approved one-at-a-time zonal placement, then the unchanged context60 pipeline."""
import json,math,os,shlex,subprocess,sys,time
from datetime import datetime
from pathlib import Path
from . import controller as c
from .context_objective import ADMISSION_V2,PLACEMENT_AUTH,authority,sha
from .context_placement_guard import KEY,PLAN,ZONES
from .context_controller import execute_pipeline
from .source_controller import prepare_ssh
from portable.venus.guard import bounded_runpod_read

R=Path('runs/venus-training-20261102/placement-v50')
SNAPSHOT='duplex-venus-context-migrate-20261102-v50'
ORIGINAL_DISK='4375908109499109167'


def cloud(*args):return c.cloud(*args)


def observe_operation(op,zone=None,seconds=900):
    until=time.time()+seconds
    while op['status']!='DONE':
        if time.time()>until:raise TimeoutError('Observe SAME operation; do not recreate or restart on timeout')
        time.sleep(5)
        op=cloud('compute','operations','describe',op['name'],*(['--zone='+zone] if zone else []))
    return op


def stockout_before_start(operation,vm):
    return 'ZONE_RESOURCE_POOL_EXHAUSTED' in json.dumps(operation.get('error',{})) and (vm is None or not vm.get('lastStartTimestamp'))


def scoped_delete_disk(name,zone,disk_id):
    actual=cloud('compute','disks','describe',name,'--zone='+zone)
    assert actual['id']==disk_id and actual['id']!=ORIGINAL_DISK and not actual.get('users')
    assert actual['sourceSnapshotId']==snapshot_id and name=='duplex-venus-context-data-20261102-'+zone[-1]+'-v50'
    c.run(['gcloud','compute','disks','delete',name,'--zone='+zone,'--project='+c.PROJECT,'--quiet'],90)


def copyback(remote_out):
    code="import pathlib,json,hashlib;r=pathlib.Path("+repr(remote_out)+");print(json.dumps([dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(r.rglob('*')) if p.is_file() and not p.is_symlink()]))"
    rows=json.loads(c.ssh('/usr/bin/python3 -c '+shlex.quote(code),90))
    c.run(['rsync','-ar','--no-owner','--no-group','--exclude=session-token','--exclude=.private','--exclude=credentials','--exclude=owner-conversations','--exclude=API keys.md','--exclude=core','--exclude=core.*','-e',shlex.join(c.ssh_argv[:-1]),c.ssh_argv[-1]+':'+remote_out+'/',str(R/'copyback')+'/'],240)
    for r in rows:
        path=R/'copyback'/r['path'];assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256']
    (R/'copyback-verified.json').write_text(json.dumps(dict(all_hashes_verified=True,files=len(rows),bytes=sum(r['bytes'] for r in rows),inventory=rows),indent=2)+'\n')
    c.record('full_context_copyback_verified',files=len(rows),bytes=sum(r['bytes'] for r in rows))


def main():
    global snapshot_id
    a=authority(ADMISSION_V2);c.R=R
    claim=R/'controller-claim.json'
    if claim.exists():raise ValueError('Observe the existing placement controller; never duplicate it')
    source=c.run(['git','rev-parse','HEAD']).strip()
    if c.run(['git','-c','core.filemode=false','status','--porcelain','--','training/venus','portable/venus','data/venus','docs/NATIVE-VENUS-CONTEXT-ZONAL-CAPACITY-AUTHORIZATION-2026-11-02.json','docs/NATIVE-VENUS-CONTEXT-TRAINING-ADMISSION-V2-2026-11-02.json']).strip():raise ValueError('Deploy committed source only')
    price=json.loads(Path('runs/venus-training-20261101/current-a100-price.json').read_text())
    assert 0<price['hourly_estimate_usd']<=4 and time.time()-datetime.fromisoformat(price['at_utc']).timestamp()<3600
    assert price['rates']['6AE1-525F-8B80']['price']<=.1
    snapshot_rates=json.loads((R/'current-snapshot-prices.json').read_text())
    assert time.time()-datetime.fromisoformat(snapshot_rates['at_utc']).timestamp()<3600
    rates={r['sku']:r for r in snapshot_rates['rates']}
    for key in ('817F-F5A3-514E','092B-DDAC-75F4','22CE-2CCB-B087','9B7C-E01E-BA5B'):
        assert rates[key]['unit']=='GiBy.mo' and rates[key]['usd']<=.05
    instances=cloud('compute','instances','list');assert all(v['status']=='TERMINATED' for v in instances)
    assert not any(v['name'].startswith('duplex-venus-context-20261102-') for v in instances)
    old=cloud('compute','disks','describe','duplex-venus-train-20261026-v1','--zone=us-central1-a')
    assert old['id']==ORIGINAL_DISK and int(old['sizeGb'])==100
    snapshot=cloud('compute','snapshots','describe',SNAPSHOT)
    assert snapshot['status']=='READY' and snapshot['sourceDiskId']==ORIGINAL_DISK and snapshot['storageLocations']==['us-central1'] and snapshot['snapshotType']=='STANDARD'
    snapshot_id=snapshot['id'];storage_deadline=datetime.fromisoformat(snapshot['creationTimestamp']).timestamp()+4*3600
    retention=json.loads((R/'snapshot-effective-retention.json').read_text())
    assert retention['retentionDurationDays'] in ('0','1','2','3')
    assert storage_deadline-time.time()>5500,'Whole GPU run and storage cleanup must fit'
    quota={v['metric']:v for v in cloud('compute','regions','describe','us-central1')['quotas']}
    assert quota['NVIDIA_A100_GPUS']['usage']==0 and quota['NVIDIA_A100_GPUS']['limit']>=1
    pods=bounded_runpod_read(Path('runs/local-migration-20261021/.private/runpod-budget-api-key').read_text().strip());pods=pods.get('pods',[]) if isinstance(pods,dict) else pods
    assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in pods)
    c.source=source;c.remote=c.ROOT+'/duplex-context50-'+source[:12]
    with claim.open('x') as f:json.dump(dict(pid=os.getpid(),source=source,snapshot_id=snapshot_id,storage_deadline_unix=storage_deadline),f,indent=2)
    c.run(['git','bundle','create',str(R/'source.bundle'),'HEAD'],120)
    awake=subprocess.Popen(['caffeinate','-i','-s','-w',str(os.getpid())])
    guard=None;vm=None;zone=None;disk=None;model_started=False;backed=False;remote_out=c.OUT+'/context-v50'
    try:
        cloud('compute','os-login','ssh-keys','add','--key-file=runs/local-migration-20261024/.private/gcp-native-ssh.pub','--ttl=2h')
        for zone in ZONES:
            assert all(v['status']=='TERMINATED' for v in cloud('compute','instances','list'))
            disk_name='duplex-venus-context-data-20261102-'+zone[-1]+'-v50'
            op=cloud('compute','disks','create',disk_name,'--zone='+zone,'--source-snapshot='+SNAPSHOT,'--size=100GB','--type=pd-balanced','--labels=duplex-scope=context-v50','--async');op=op[0] if isinstance(op,list) else op
            c.record('one_restored_disk_operation',zone=zone,operation=op['name'])
            op=observe_operation(op,zone)
            if op.get('error'):raise RuntimeError(json.dumps(op['error']))
            disk=cloud('compute','disks','describe',disk_name,'--zone='+zone)
            assert int(disk['sizeGb'])==100 and disk['sourceSnapshotId']==snapshot_id and not disk.get('users')
            (R/('disk-'+zone+'.json')).write_text(json.dumps(disk,indent=2)+'\n')
            c.NAME='duplex-venus-context-20261102-'+zone[-1]+'-v50'
            bootstrap=R/'bootstrap.sh';bootstrap.write_text('#!/bin/sh\nset -eu\nsystemd-run --unit=duplex-context50-bootstrap-stop --on-active=600s /sbin/shutdown -h now || /sbin/shutdown -h now\n')
            op=cloud('compute','instances','create',c.NAME,'--zone='+zone,'--machine-type=a2-highgpu-1g','--provisioning-model=STANDARD','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+disk_name+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(bootstrap),'--max-run-duration=3900s','--instance-termination-action=STOP','--async');op=op[0] if isinstance(op,list) else op
            c.record('one_approved_gpu_attempt',zone=zone,operation=op['name'],source=source)
            op=observe_operation(op,zone)
            (R/('gpu-operation-'+zone+'.json')).write_text(json.dumps(op,indent=2)+'\n')
            found=[v for v in cloud('compute','instances','list') if v['name']==c.NAME]
            vm=cloud('compute','instances','describe',c.NAME,'--zone='+zone) if found else None
            if op.get('error'):
                if not stockout_before_start(op,vm):raise RuntimeError('Not a pre-start stockout; no replacement retry: '+json.dumps(op['error']))
                if vm:
                    assert vm['status']=='TERMINATED'
                    c.run(['gcloud','compute','instances','delete',c.NAME,'--zone='+zone,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],90)
                vm=None;scoped_delete_disk(disk_name,zone,disk['id']);disk=None
                c.record('terminal_stockout_no_gpu_or_model_start',zone=zone)
                continue
            assert vm and vm['status']=='RUNNING'
            break
        else:raise RuntimeError('All three approved zones returned terminal stockout; no model launched')
        start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+3900
        assert end+180<storage_deadline
        (R/'provider-start.json').write_text(json.dumps(vm,indent=2)+'\n')
        prepare_ssh(vm,zone,'50');c.sync(R/'source.bundle',c.ROOT+'/context50.bundle')
        c.ssh('git --git-dir='+c.ROOT+'/source.git fetch '+c.ROOT+'/context50.bundle HEAD && git --git-dir='+c.ROOT+'/source.git worktree add --detach '+c.remote+' '+source)
        plan=dict(resource_key=KEY,authorization_sha256=sha(PLACEMENT_AUTH),new_optimizer_updates_authorized_by_this_execution=60,maximum_gpu_seconds=3900,maximum_new_gpu_running_intervals=1,vm_name=c.NAME,vm_id=vm['id'],disk_name=disk['name'],disk_id=disk['id'],zone=zone,original_disk_id=ORIGINAL_DISK,snapshot_source_disk_id=snapshot['sourceDiskId'],snapshot_id=snapshot_id,snapshot_name=SNAPSHOT)
        PLAN.write_text(json.dumps(plan,indent=2)+'\n');c.ssh('mkdir -p '+c.remote+'/'+str(PLAN.parent));c.sync(PLAN,c.remote+'/'+str(PLAN))
        lease=dict(resource_key=KEY,zone=zone,vm_id=vm['id'],disk_id=disk['id'],disk_size_gib=100,snapshot_id=snapshot_id,creation_timestamp=vm['creationTimestamp'],stop_unix=end,storage_deadline_unix=storage_deadline,prior_new_gpu_seconds_upper=28700,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=price['hourly_estimate_usd'],authorization_sha256=sha(PLACEMENT_AUTH),execution_plan_sha256=sha(PLAN),controller_pid=os.getpid(),controller_birth=c.run(['/bin/ps','-p',str(os.getpid()),'-o','lstart=']).strip())
        lp=c.ROOT+'/context50-lease.json';(R/'lease.json').write_text(json.dumps(lease,indent=2)+'\n');c.sync(R/'lease.json',lp)
        guard=subprocess.Popen([sys.executable,'-u','-m','portable.venus.guard','--lease',str(R/'lease.json'),'--ssh-argv',str(R/'ssh.json'),'--runpod-key','runs/local-migration-20261021/.private/runpod-budget-api-key','--status',str(R/'guard.json'),'--stop-file',str(R/'stop-requested')],stdout=(R/'guard.log').open('x'),stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});c.guard=guard
        until=time.time()+90
        while not (R/'guard.json').exists():
            if guard.poll() is not None or time.time()>until:raise RuntimeError('Independent provider guard not live')
            time.sleep(1)
        live=json.loads(c.ssh('cat '+c.ROOT+'/venus-live-guard-v1.json'));assert live['vm_id']==vm['id'] and live['stop_unix']==end and time.time()<live['expires_unix']
        c.ssh('sudo systemd-run --unit=duplex-context50-guest --property=RuntimeMaxSec=4000 --working-directory='+c.remote+' /usr/bin/python3 -u -m portable.venus.guard --guest --guard '+c.ROOT+'/venus-live-guard-v1.json --lease '+lp)
        assert c.props('duplex-context50-guest')['ActiveState']=='active';c.ssh('sudo systemctl stop duplex-context50-bootstrap-stop.timer')
        c.record('both_guards_live_before_model',mac_pid=guard.pid,vm_id=vm['id'],disk_id=disk['id'],native_stop_unix=end)
        assert int(c.ssh('df -B1 --output=avail /workspace/migration | tail -1').strip())>8*1024**3
        assert c.ssh('sha256sum '+c.OUT+'/action-candidate-v42/step-0108.pt').split()[0]==a['parent_checkpoint_sha256']
        assert c.ssh('sha256sum '+c.OUT+'/paced-prefix-cache-v24/manifest.json').split()[0]==a['old_prefix_manifest_sha256']
        c.ssh('mkdir -p '+remote_out)
        for local,dest in ((a['new_timeline_path']+'/',remote_out+'/timelines/'),('runs/venus-training-20261101/context-alignment-v1/',remote_out+'/alignment/'),('runs/venus-training-20261101/context-semantic-v1/',remote_out+'/semantic/')):c.sync(local,dest)
        c.sync('runs/venus-training-20261029/action-target-audit-v1/index-verified-v2.json',remote_out+'/old-action-index.json')
        model_started=True
        execute_pipeline(remote_out,end,lp,a,c.remote+'/'+str(ADMISSION_V2),'duplex-venus-context50')
    except BaseException as error:
        c.record('placement_or_training_failure',error_type=type(error).__name__,error=str(error)[-1800:]);raise
    finally:
        if c.ssh_argv and model_started:
            try:
                for unit in c.UNITS:
                    if c.props(unit).get('ActiveState')=='active':c.ssh('sudo systemctl stop '+unit,30)
                copyback(remote_out);backed=True
            except Exception as error:c.record('copyback_failed_preserve_new_disk',error_type=type(error).__name__,error=str(error)[-600:])
        if vm and zone:
            current=cloud('compute','instances','describe',c.NAME,'--zone='+zone)
            if current['status'] not in ('TERMINATED','STOPPED'):c.run(['gcloud','compute','instances','stop',c.NAME,'--zone='+zone,'--project='+c.PROJECT,'--quiet','--discard-local-ssd=True'],150)
            current=cloud('compute','instances','describe',c.NAME,'--zone='+zone);(R/'provider-final.json').write_text(json.dumps(current,indent=2)+'\n')
            if current.get('lastStartTimestamp'):
                actual=datetime.fromisoformat(current['lastStopTimestamp']).timestamp()-datetime.fromisoformat(current['lastStartTimestamp']).timestamp()
                c.record('actual_gpu_stopped',actual_seconds=actual,upper_seconds=math.ceil(actual/100)*100,prior_focused_seconds=28700)
            if backed or not model_started:
                c.run(['gcloud','compute','instances','delete',c.NAME,'--zone='+zone,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],90)
                vm=None
        if disk and not vm and (backed or not model_started):
            scoped_delete_disk(disk['name'],zone,disk['id']);disk=None
            c.record('temporary_clone_removed_original_preserved')
        if not disk:
            s=cloud('compute','snapshots','describe',SNAPSHOT);assert s['id']==snapshot_id and s['sourceDiskId']==ORIGINAL_DISK
            c.run(['gcloud','compute','snapshots','delete',SNAPSHOT,'--project='+c.PROJECT,'--quiet'],90)
            c.record('temporary_snapshot_removed_recycle_cost_reserved')
        if guard and guard.poll() is None:
            try:guard.wait(timeout=35)
            except subprocess.TimeoutExpired:guard.terminate()
        awake.terminate()


if __name__=='__main__':main()
