"""One longer sequential A100 lease under the existing focused-repair grant."""
import hashlib,json
from pathlib import Path
from datetime import datetime
from .focused_guard import AUTH
PLAN=Path('docs/NATIVE-VENUS-PACED-CONTINUATION-EXECUTION-2026-10-27.json')
PLAN_V23=Path('docs/NATIVE-VENUS-PACED-WARM-RECOVERY-EXECUTION-2026-10-27.json')
PLAN_V24=Path('docs/NATIVE-VENUS-PACED-503-RECOVERY-EXECUTION-2026-10-27.json')

def execution_plan(lease):return PLAN_V24 if lease.get('resource_key')=='venus-paced-recovery-v24' else PLAN_V23 if lease.get('resource_key')=='venus-paced-continuation-v23' else PLAN

def resource(lease):
    path=execution_plan(lease);raw=AUTH.read_bytes();a=json.loads(raw);p=json.loads(path.read_text());r=a['resources']
    assert a['status']=='APPROVED' and a['human_approval']=='Approve $40 / four GPU hours'
    assert a['maximum_additional_gcp_usage_usd']==40 and a['maximum_additional_gpu_seconds']==14400
    assert p['status']=='WITHIN_APPROVED_FOCUSED_REPAIR' and p['parent_sha256']==hashlib.sha256(raw).hexdigest()
    assert lease['authorization_sha256']==hashlib.sha256(raw).hexdigest() and lease['execution_plan_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert lease['resource_key'] in ('venus-paced-continuation-v22','venus-paced-continuation-v23','venus-paced-recovery-v24') and lease['zone']==r['zone'] and lease['disk_id']==r['same_existing_disk_id']
    assert str(lease['vm_id']).isdigit() and lease['disk_size_gib']==100
    if lease['resource_key']=='venus-paced-continuation-v23':assert str(lease['vm_id'])==p['retire_only_vm_id']=='2800910655359330463'
    return dict(name=p['vm_name'],zone=r['zone'],ident=str(lease['vm_id']),disk=r['same_existing_disk_id'],disk_name=r['same_existing_disk_name'],machine='a2-highgpu-1g',authority=AUTH)

def validate(vm,instances,pods,lease,now):
    r=resource(lease);p=json.loads(execution_plan(lease).read_text());start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();duration=p['maximum_gpu_seconds'];end=start+duration
    assert str(vm['id'])==r['ident'] and vm['name']==r['name'] and vm['status']=='RUNNING'
    assert vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone']) and vm['creationTimestamp']==lease['creation_timestamp']
    assert vm['tags']['items']==['duplex-venus-train-bounded'] and not vm.get('serviceAccounts')
    assert any(d.get('boot') and not d.get('autoDelete') and d['source'].endswith('/disks/'+r['disk_name']) for d in vm['disks'])
    s=vm['scheduling'];assert s['provisioningModel']=='FLEX_START' and int(s['maxRunDuration']['seconds'])==duration
    assert s['instanceTerminationAction']=='STOP' and not s['automaticRestart'] and start<=now<end and lease['stop_unix']==end
    prior=p['focused_gpu_seconds_upper_before_launch']
    expected=(6500,7200) if lease['resource_key']=='venus-paced-recovery-v24' else (2600,10000) if lease['resource_key']=='venus-paced-continuation-v23' else (2400,10000)
    assert (prior,duration)==expected
    assert lease['prior_gpu_seconds']==54451 and lease['prior_new_gpu_seconds_upper']==prior and lease['prior_scope_usd']==91.98448055555555
    assert 0<lease['verified_hourly_price_usd']<=4 and prior+duration<=14400 and 54451+prior+duration<=72000
    assert (prior+duration)*4/3600+10+6+8<=40
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ['a2-','a3-','a4-','g2-']))]
    assert len(running)==1 and str(running[0]['id'])==r['ident'] and all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    return end
