"""One approved replacement GPU interval, actual cloned-disk/provider identities."""
import json
from pathlib import Path
from datetime import datetime
from .context_objective import CONTINUATION_AUTH as PLACEMENT_AUTH,ADMISSION_V3 as ADMISSION_V2,authority,sha

KEY='venus-context-train-v52'
PLAN=Path('runs/venus-training-20261102/continuation-v52/execution.json')
ZONES=('us-central1-c','us-central1-f','us-central1-b')


def execution_plan(lease):
    assert lease['resource_key']==KEY
    return PLAN


def resource(lease):
    authority(ADMISSION_V2)
    a=json.loads(PLACEMENT_AUTH.read_text());p=json.loads(PLAN.read_text())
    assert lease['resource_key']==KEY and lease['authorization_sha256']==sha(PLACEMENT_AUTH)
    assert p['authorization_sha256']==sha(PLACEMENT_AUTH) and lease['execution_plan_sha256']==sha(PLAN)
    assert p['new_optimizer_updates_authorized_by_this_execution']==60 and p['maximum_gpu_seconds']==6600
    assert p['original_disk_id']=='4375908109499109167' and p['snapshot_source_disk_id']==p['original_disk_id']
    assert lease['zone']==p['zone'] and lease['zone'] in ZONES
    assert p['vm_name']=='duplex-venus-context-20261102-'+lease['zone'][-1]+'-v52'
    assert p['disk_name']=='duplex-venus-context-data-20261102-'+lease['zone'][-1]+'-v52'
    assert p['snapshot_name']=='duplex-venus-context-migrate-20261102-v50'
    assert str(lease['vm_id'])==p['vm_id'] and lease['disk_id']==p['disk_id'] and lease['snapshot_id']==p['snapshot_id']
    assert all(str(lease[k]).isdigit() for k in ('vm_id','disk_id','snapshot_id')) and lease['disk_id']!=p['original_disk_id']
    assert lease['disk_size_gib']==100 and p['maximum_new_gpu_running_intervals']==1
    return dict(name=p['vm_name'],zone=p['zone'],ident=p['vm_id'],disk=p['disk_id'],disk_name=p['disk_name'],machine='a2-highgpu-1g',authority=PLACEMENT_AUTH,provider_read_limits=(25,20))


def validate(vm,instances,pods,lease,now):
    r=resource(lease);p=json.loads(PLAN.read_text())
    start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+6600
    assert str(vm['id'])==r['ident'] and vm['name']==r['name'] and vm['status']=='RUNNING'
    assert vm['machineType'].endswith('/a2-highgpu-1g') and vm['zone'].endswith('/'+r['zone'])
    assert vm['creationTimestamp']==lease['creation_timestamp']
    assert set(vm['tags']['items'])=={'duplex-venus-train-bounded'} and not vm.get('serviceAccounts')
    assert len(vm['disks'])==1 and vm['disks'][0]['boot'] and not vm['disks'][0]['autoDelete']
    assert int(vm['disks'][0]['diskSizeGb'])==100 and vm['disks'][0]['source'].endswith('/disks/'+r['disk_name'])
    s=vm['scheduling'];assert s['provisioningModel']=='STANDARD' and int(s['maxRunDuration']['seconds'])==6600
    assert s['instanceTerminationAction']=='STOP' and not s['automaticRestart']
    assert start<=now<end and lease['stop_unix']==end
    assert lease['prior_new_gpu_seconds_upper']==30000 and lease['prior_gpu_seconds']==54451 and lease['prior_scope_usd']==91.98448055555555
    assert 0<lease['verified_hourly_price_usd']<=4
    assert 30000+6700<=36700 and 54451+30000+6700<=91151
    assert (30000+6700)*4/3600+33+1<=75
    assert 91.98448055555555+(30000+6700)*4/3600+33+1<=175
    assert now<lease['storage_deadline_unix'] and end<lease['storage_deadline_unix']
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ('a2-','a3-','a4-','g2-')))]
    assert len(running)==1 and str(running[0]['id'])==r['ident']
    assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in pods)
    return end
