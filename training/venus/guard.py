"""New conditional-training authority; old inference leases remain immutable."""
import hashlib
import json
from pathlib import Path
from datetime import datetime

AUTH=Path('docs/NATIVE-VENUS-CONTROL-TRAINING-AUTHORIZATION-2026-10-26.json')


def resource(lease):
    raw=AUTH.read_bytes();a=json.loads(raw);r=a['resources']
    assert a['status']=='APPROVED' and a['maximum_additional_gcp_usage_usd']==35
    assert a['maximum_additional_gpu_seconds']==14400 and a['prior_new_scope_gpu_seconds_upper']==4600
    assert a['maximum_single_allocation_seconds']==7200 and a['total_new_scope_conservative_upper_usd']<35
    assert r['maximum_concurrent_gpu_jobs']==1 and r['maximum_temporary_boot_disk_gib']==100
    assert lease['authorization_sha256']==hashlib.sha256(raw).hexdigest()
    assert lease['zone'] in r['zones'] and all(str(lease[k]).isdigit() for k in ('vm_id','disk_id'))
    return dict(name=r['name'],disk_name=r['disk_name'],zone=lease['zone'],ident=str(lease['vm_id']),disk=str(lease['disk_id']),machine=r['machine'],authority=AUTH)


def stamp(value):return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()


def validate(vm,instances,pods,lease,now):
    r=resource(lease);a=json.loads(AUTH.read_text())
    assert str(vm['id'])==r['ident'] and vm['name']==r['name'] and vm['status']=='RUNNING'
    assert vm['zone'].endswith('/'+r['zone']) and vm['machineType'].endswith('/'+r['machine'])
    assert vm['creationTimestamp']==lease['creation_timestamp']
    assert any(d.get('boot') and not d.get('autoDelete') and d['source'].endswith('/disks/'+r['disk_name']) for d in vm['disks'])
    assert vm['tags']['items']==['duplex-venus-train-bounded']
    s=vm['scheduling'];start=stamp(vm['lastStartTimestamp']);end=start+7200
    assert s['provisioningModel']=='FLEX_START' and int(s['maxRunDuration']['seconds'])==7200
    assert s['instanceTerminationAction']=='STOP' and not s['automaticRestart']
    assert start<=now<end and lease['stop_unix']==end
    assert lease['prior_new_gpu_seconds_upper']==4600 and lease['prior_gpu_seconds']==40351
    assert lease['prior_scope_usd']==a['prior_original_scope_cost_upper_usd'] and lease['disk_size_gib']==100
    assert 4600+7200<=14400 and 40351+4600+7200<=57600
    assert (4600+7200)*7.25/3600+6+4<35
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ['a2-','a3-','a4-','g2-']))]
    assert len(running)==1 and str(running[0]['id'])==r['ident']
    assert all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    return end
