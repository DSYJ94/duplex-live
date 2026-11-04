"""Scoped approved snapshot reuse and pre-start-only zonal capacity fallback."""
import json,subprocess,time,urllib.request,urllib.error
from pathlib import Path
from datetime import datetime,timezone
from . import controller as c
ORIGINAL='4375908109499109167'
SNAPSHOT='duplex-venus-context-migrate-20261102-v50'


def save(name,value):
    (c.R/name).write_text(json.dumps(value,indent=2)+'\n')


def api(path,body=None):
    token=c.run(['gcloud','auth','print-access-token']).strip()
    req=urllib.request.Request('https://compute.googleapis.com/compute/beta/projects/'+c.PROJECT+'/global/'+path,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'},data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req,timeout=45) as f:return json.load(f)


def observe(op,zone=None):
    until=time.time()+900
    while op['status']!='DONE':
        assert time.time()<until,'Observe SAME operation; never resubmit'
        time.sleep(4)
        op=c.cloud('compute','operations','describe',op['name'],*(['--zone='+zone] if zone else []))
        save('latest-operation.json',op)
    return op


def owned_copies():
    values=api('recoverableSnapshots');save('recoverable-snapshots.json',values)
    rows=[v for v in values.get('items',[]) if v.get('originalResource',{}).get('sourceDiskId')==ORIGINAL]
    for v in rows:
        assert (datetime.fromisoformat(v['purgeTimestamp'])-datetime.fromisoformat(v['creationTimestamp'])).total_seconds()<=72*3600+2
    return rows


def snapshot():
    # Official current storage SKUs, not a remembered price.
    wanted={'817F-F5A3-514E','092B-DDAC-75F4','22CE-2CCB-B087','9B7C-E01E-BA5B'};found={};page=''
    token=c.run(['gcloud','auth','print-access-token']).strip()
    for _ in range(20):
        url='https://cloudbilling.googleapis.com/v1/services/6F81-5844-456A/skus?currencyCode=USD&pageSize=5000'+('&pageToken='+page if page else '')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':'Bearer '+token}),timeout=60) as f:data=json.load(f)
                break
            except urllib.error.HTTPError as error:
                if error.code!=429 or attempt==2:raise
                # Only retry this read-only catalog page, never a resource create.
                time.sleep(20*(attempt+1))
        for row in data.get('skus',[]):
            if row['skuId'] in wanted:
                p=row['pricingInfo'][-1]['pricingExpression'];v=p['tieredRates'][0]['unitPrice'];usd=int(v.get('units',0))+v.get('nanos',0)/1e9
                assert p['usageUnit']=='GiBy.mo' and usd<=.05
                found[row['skuId']]=dict(unit=p['usageUnit'],usd=usd,description=row['description'])
        if set(found)==wanted:break
        page=data.get('nextPageToken','')
        if not page:break
    assert set(found)==wanted;save('snapshot-prices.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),rates=found))
    policy=api('snapshotRecycleBinPolicy');save('snapshot-policy.json',policy)
    def durations(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if key=='retentionDurationDays':yield int(item)
                else:yield from durations(item)
        elif isinstance(value,list):
            for item in value:yield from durations(item)
    retention=list(durations(policy));assert retention and max(retention)<=3
    rows=owned_copies();active=[s for s in c.cloud('compute','snapshots','list') if s.get('sourceDiskId')==ORIGINAL]
    assert len(active)<=1
    assert sum(int(v['originalResource']['storageBytes']) for v in rows)+sum(int(v['storageBytes']) for v in active)<=160*1024**3
    if active:
        snap=active[0];assert snap['name']==SNAPSHOT and snap['id']=='4724490043241393447'
    else:
        compatible=[v for v in rows if v['originalResource']['name']==SNAPSHOT and v['originalResource']['id']=='4724490043241393447']
        assert compatible,'No compatible owned snapshot: stop before broader storage changes'
        chosen=compatible[-1]
        assert sum(int(v['originalResource']['storageBytes']) for v in rows)+int(chosen['originalResource']['storageBytes'])<=160*1024**3
        with (c.R/'snapshot-recover-claim.json').open('x') as f:json.dump(dict(recoverable_id=chosen['id'],name=chosen['name']),f)
        op=api('recoverableSnapshots/'+chosen['name']+'/recover',{});save('snapshot-recover-operation.json',op);c.record('one_owned_snapshot_recovery',operation=op['name'])
        op=observe(op);save('snapshot-recover-terminal.json',op);assert not op.get('error'),op.get('error')
        snap=c.cloud('compute','snapshots','describe',SNAPSHOT)
    assert snap['status']=='READY' and snap['sourceDiskId']==ORIGINAL and snap['storageLocations']==['us-central1'] and snap['snapshotType']=='STANDARD' and int(snap['diskSizeGb'])==100
    rule=api('snapshots/'+snap['name']+'/getEffectiveRecycleBinRule');save('snapshot-effective-retention.json',rule);assert int(rule['retentionDurationDays'])<=3
    rows=owned_copies();assert sum(int(v['originalResource']['storageBytes']) for v in rows)+int(snap['storageBytes'])<=160*1024**3
    save('snapshot-ready.json',snap);return snap


def stockout_before_start(op,vm):
    return 'ZONE_RESOURCE_POOL_EXHAUSTED' in json.dumps(op.get('error',{})) and (vm is None or not vm.get('lastStartTimestamp'))


def retire(vm,disk,zone,snap,version='56'):
    assert version in ('56','57','58','59')
    if vm:
        actual=c.cloud('compute','instances','describe',vm['name'],'--zone='+zone)
        assert actual['id']==vm['id'] and actual['status']=='TERMINATED' and actual['name']=='duplex-venus-tts-20261102-'+zone[-1]+'-v'+version
        assert actual['disks'][0]['source'].endswith('/disks/'+disk['name'])
        c.run(['gcloud','compute','instances','delete',vm['name'],'--zone='+zone,'--project='+c.PROJECT,'--keep-disks=boot','--quiet'],150)
    actual=c.cloud('compute','disks','describe',disk['name'],'--zone='+zone)
    assert actual['id']==disk['id'] and actual['id']!=ORIGINAL and actual['sourceSnapshotId']==snap['id'] and not actual.get('users')
    assert actual['name']=='duplex-venus-tts-data-20261102-'+zone[-1]+'-v'+version
    c.run(['gcloud','compute','disks','delete',disk['name'],'--zone='+zone,'--project='+c.PROJECT,'--quiet'],150)
    c.record('temporary_envelope_retired',vm_id=vm['id'] if vm else None,disk_id=disk['id'],zone=zone)


def retire_snapshot(snap):
    actual=c.cloud('compute','snapshots','describe',snap['name']);assert actual['id']==snap['id'] and actual['sourceDiskId']==ORIGINAL and actual['name']==SNAPSHOT
    c.run(['gcloud','compute','snapshots','delete',snap['name'],'--project='+c.PROJECT,'--quiet'],150)
    c.record('active_owned_snapshot_retired',snapshot_id=snap['id'],recycle_retention_reserved=True)


def allocate(version='56'):
    assert version in ('56','57','58','59')
    duration=1800 if version=='56' else 1200
    if version=='59':
        from .tts_completion_capacity_guard import AUTH,sha
        a=json.loads(AUTH.read_text());assert a['status']=='APPROVED' and sha(a['proposal_path'])==a['proposal_sha256']
        p=json.loads(Path(a['proposal_path']).read_text());assert p['compute']['ordered_zones']==['us-central1-c','us-central1-b']
        assert p['storage']['new_temporary_window_seconds']==7200
        failed=p['actual_failed_attempt'];assert sha(failed['events_path'])==failed['events_sha256']
        assert failed['gpu_intervals']==failed['model_runs']==0
        assert p['storage']['clone_aggregate_upper_GiBh']<=400 and p['storage']['active_snapshot_aggregate_upper_GiBh']<=400
        deadline=time.time()+7200
    elif version=='58':
        from .tts_completion_window_guard import AUTH,sha
        a=json.loads(AUTH.read_text());assert a['status']=='APPROVED' and sha(a['proposal_path'])==a['proposal_sha256']
        p=json.loads(Path(a['proposal_path']).read_text());assert p['storage']['temporary_window_seconds']==7200
        proof=p['storage']['prior_storage_proof']
        assert sha(proof['events_path'])==proof['events_sha256'] and sha(proof['disk_receipt_path'])==proof['disk_receipt_sha256']
        assert p['storage']['prior_v56_clone_GiBh_upper']+100*7200/3600<=400
        assert p['storage']['prior_v56_active_snapshot_GiBh_upper']+100*7200/3600<=400
        deadline=time.time()+7200
    else:deadline=time.time()+4*3600 if version=='56' else json.loads((c.R.parent/'tts-placement-v56/storage-deadline.json').read_text())['storage_deadline_unix']
    assert time.time()+2400<deadline,'Expired or insufficient storage window; no snapshot/clone mutation'
    save('storage-deadline.json',dict(storage_deadline_unix=deadline))
    snap=snapshot()
    # The whole loop owns one disk/envelope at a time; never retry a real interval.
    for zone in (('us-central1-c','us-central1-b') if version=='59' else ('us-central1-f','us-central1-c','us-central1-b') if version=='56' else ('us-central1-f',)):
        assert time.time()+2400<deadline
        assert all(v['status']=='TERMINATED' for v in c.cloud('compute','instances','list'))
        name='duplex-venus-tts-20261102-'+zone[-1]+'-v'+version;disk_name='duplex-venus-tts-data-20261102-'+zone[-1]+'-v'+version
        assert not any(v['name'].startswith('duplex-venus-tts-data-20261102-') for v in c.cloud('compute','disks','list'))
        c.record('one_clone_request',name=disk_name,zone=zone)
        c.run(['gcloud','compute','disks','create',disk_name,'--zone='+zone,'--source-snapshot='+snap['name'],'--size=100GB','--type=pd-balanced','--labels=duplex-scope=tts-v'+version,'--project='+c.PROJECT,'--quiet','--format=json'],900)
        disk=c.cloud('compute','disks','describe',disk_name,'--zone='+zone);save('disk-'+zone+'.json',disk)
        assert disk['sourceSnapshotId']==snap['id'] and int(disk['sizeGb'])==100 and not disk.get('users')
        c.NAME=name
        op=c.cloud('compute','instances','create',name,'--zone='+zone,'--machine-type=a2-highgpu-1g','--provisioning-model=STANDARD','--maintenance-policy=TERMINATE','--no-restart-on-failure','--no-service-account','--no-scopes','--tags=duplex-venus-train-bounded','--disk=name='+disk_name+',boot=yes,auto-delete=no,mode=rw','--metadata=enable-oslogin=TRUE,enable-guest-attributes=TRUE,block-project-ssh-keys=TRUE','--metadata-from-file=startup-script='+str(c.R/'bootstrap.sh'),'--max-run-duration='+str(duration)+'s','--instance-termination-action=STOP','--async')
        op=op[0] if isinstance(op,list) else op;save('create-'+zone+'.json',op);c.record('one_tts_placement_attempt',zone=zone,operation=op['name'])
        op=observe(op,zone);save('terminal-'+zone+'.json',op)
        matches=[v for v in c.cloud('compute','instances','list') if v['name']==name];vm=matches[0] if matches else None
        if op.get('error'):
            if not stockout_before_start(op,vm):raise RuntimeError('No further attempts authorized: '+json.dumps(op.get('error')))
            retire(vm,disk,zone,snap,version);continue
        assert vm and vm['status']=='RUNNING' and vm.get('lastStartTimestamp')
        save('allocation.json',dict(vm=vm,disk=disk,zone=zone,snapshot=snap,storage_deadline_unix=deadline))
        return vm,disk,zone,snap,deadline
    retire_snapshot(snap)
    raise RuntimeError('All approved zones stockout before start; zero model/GPU intervals')
