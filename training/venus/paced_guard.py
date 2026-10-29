"""One longer sequential A100 lease under the existing focused-repair grant."""
import hashlib,json,math
from pathlib import Path
from datetime import datetime
from .focused_guard import AUTH
PLAN=Path('docs/NATIVE-VENUS-PACED-CONTINUATION-EXECUTION-2026-10-27.json')
PLAN_V23=Path('docs/NATIVE-VENUS-PACED-WARM-RECOVERY-EXECUTION-2026-10-27.json')
PLAN_V24=Path('docs/NATIVE-VENUS-PACED-503-RECOVERY-EXECUTION-2026-10-27.json')
PLAN_V25=Path('docs/NATIVE-VENUS-PREPARED-REACTIVE-EXECUTION-2026-10-28.json')
PLAN_V26=Path('docs/NATIVE-VENUS-PAIRED-LOADER-REPAIR-EXECUTION-2026-10-28.json')
PLAN_V27=Path('docs/NATIVE-VENUS-LIVE-TRIAL-EXECUTION-2026-10-28.json')
PLAN_V28=Path('docs/NATIVE-VENUS-LIVE-WARM-RETRY-EXECUTION-2026-10-28.json')
PLAN_V29=Path('docs/NATIVE-VENUS-REPAIRED-LIVE-EXECUTION-2026-10-28.json')
PLAN_V30=Path('docs/NATIVE-VENUS-STABLE-LIVE-EXECUTION-2026-10-28.json')
PLAN_V31=Path('docs/NATIVE-VENUS-OWNER-RETRY-EXECUTION-2026-10-28.json')
PLAN_V32=Path('docs/NATIVE-VENUS-LIVE-REOPEN-EXECUTION-2026-10-28.json')
PLAN_V33=Path('docs/NATIVE-VENUS-REMAINING-PROFILE-EXECUTION-2026-10-29.json')
REOPEN_AUTH=Path('docs/NATIVE-VENUS-OWNER-LIVE-REOPEN-AUTHORIZATION-2026-10-28.json')
LIVE_AUTH=Path('docs/NATIVE-VENUS-LIVE-TIME-EXTENSION-AUTHORIZATION-2026-10-28.json')

def execution_plan(lease):
    if lease.get('resource_key') in ('venus-adaptive-v34','venus-adaptive-v35','venus-adaptive-v36'):
        from .adaptive_guard import execution_plan as adaptive_plan
        return adaptive_plan(lease)
    return PLAN_V33 if lease.get('resource_key')=='venus-paced-recovery-v33' else PLAN_V32 if lease.get('resource_key')=='venus-paced-recovery-v32' else PLAN_V31 if lease.get('resource_key')=='venus-paced-recovery-v31' else PLAN_V30 if lease.get('resource_key')=='venus-paced-recovery-v30' else PLAN_V29 if lease.get('resource_key')=='venus-paced-recovery-v29' else PLAN_V28 if lease.get('resource_key')=='venus-paced-recovery-v28' else PLAN_V27 if lease.get('resource_key')=='venus-paced-recovery-v27' else PLAN_V26 if lease.get('resource_key')=='venus-paced-recovery-v26' else PLAN_V25 if lease.get('resource_key')=='venus-paced-recovery-v25' else PLAN_V24 if lease.get('resource_key')=='venus-paced-recovery-v24' else PLAN_V23 if lease.get('resource_key')=='venus-paced-continuation-v23' else PLAN

def resource(lease):
    path=execution_plan(lease);authority=REOPEN_AUTH if lease.get('resource_key') in ('venus-paced-recovery-v32','venus-paced-recovery-v33') else LIVE_AUTH if lease.get('resource_key') in ('venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33') else AUTH
    raw=authority.read_bytes();a=json.loads(raw);p=json.loads(path.read_text());r=a['resources']
    assert a['status']=='APPROVED'
    if authority==REOPEN_AUTH:
        assert a['human_approval']=='Proceed with next steps as well' and a['maximum_additional_gcp_usage_usd']==70 and a['maximum_additional_gpu_seconds']==18600
        assert a['parent_sha256']==hashlib.sha256(LIVE_AUTH.read_bytes()).hexdigest() and a['new_optimizer_updates']==0
    elif authority==LIVE_AUTH:
        assert a['human_approval']=='Cap of $70' and a['maximum_additional_gcp_usage_usd']==70 and a['maximum_additional_gpu_seconds']==17300
        assert a['parent_sha256']==hashlib.sha256(AUTH.read_bytes()).hexdigest() and a['new_optimizer_updates']==0
    else:
        assert a['human_approval']=='Approve $40 / four GPU hours'
        assert a['maximum_additional_gcp_usage_usd']==40 and a['maximum_additional_gpu_seconds']==14400
    assert p['status']=='WITHIN_APPROVED_FOCUSED_REPAIR' and p['parent_sha256']==hashlib.sha256(raw).hexdigest()
    assert lease['authorization_sha256']==hashlib.sha256(raw).hexdigest() and lease['execution_plan_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert lease['resource_key'] in ('venus-paced-continuation-v22','venus-paced-continuation-v23','venus-paced-recovery-v24','venus-paced-recovery-v25','venus-paced-recovery-v26','venus-paced-recovery-v27','venus-paced-recovery-v28','venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33') and lease['zone']==r['zone'] and lease['disk_id']==r['same_existing_disk_id']
    assert str(lease['vm_id']).isdigit() and lease['disk_size_gib']==100
    if lease['resource_key']=='venus-paced-continuation-v23':assert str(lease['vm_id'])==p['retire_only_vm_id']=='2800910655359330463'
    return dict(name=p['vm_name'],zone=r['zone'],ident=str(lease['vm_id']),disk=r['same_existing_disk_id'],disk_name=r['same_existing_disk_name'],machine='a2-highgpu-1g',authority=authority)

def validate(vm,instances,pods,lease,now):
    r=resource(lease);p=json.loads(execution_plan(lease).read_text());start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();duration=p['maximum_gpu_seconds'];end=start+duration
    assert str(vm['id'])==r['ident'] and vm['name']==r['name'] and vm['status']=='RUNNING'
    assert vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone']) and vm['creationTimestamp']==lease['creation_timestamp']
    assert vm['tags']['items']==['duplex-venus-train-bounded'] and not vm.get('serviceAccounts')
    assert any(d.get('boot') and not d.get('autoDelete') and d['source'].endswith('/disks/'+r['disk_name']) for d in vm['disks'])
    s=vm['scheduling'];assert s['provisioningModel']=='FLEX_START' and int(s['maxRunDuration']['seconds'])==duration
    assert s['instanceTerminationAction']=='STOP' and not s['automaticRestart'] and start<=now<end and lease['stop_unix']==end
    prior=p['focused_gpu_seconds_upper_before_launch']
    expected=(10500,3900) if lease['resource_key']=='venus-paced-recovery-v26' else (9900,4500) if lease['resource_key']=='venus-paced-recovery-v25' else (6500,7200) if lease['resource_key']=='venus-paced-recovery-v24' else (2600,10000) if lease['resource_key']=='venus-paced-continuation-v23' else (2400,10000)
    if lease['resource_key']=='venus-paced-recovery-v27':
        last=p['previous_provider_snapshot'];assert last['status']=='TERMINATED' and str(last['id'])=='6750796068068004320'
        assert start>=datetime.fromisoformat(last['lastStopTimestamp']).timestamp()
        elapsed=datetime.fromisoformat(last['lastStopTimestamp']).timestamp()-datetime.fromisoformat(last['lastStartTimestamp']).timestamp()
        expected_prior=10500+math.ceil(elapsed/100)*100
        expected=(expected_prior,min(900,14400-expected_prior));assert expected[1]>=600
    if lease['resource_key']=='venus-paced-recovery-v28':
        assert str(vm['id'])==p['retire_only_vm_id']=='5521472841931556416'
        expected=(13200,900)
    if lease['resource_key']=='venus-paced-recovery-v29':expected=(13700,3600)
    if lease['resource_key']=='venus-paced-recovery-v30':expected=(14000,3300)
    if lease['resource_key']=='venus-paced-recovery-v31':expected=(14600,2700)
    if lease['resource_key']=='venus-paced-recovery-v32':expected=(16800,1800)
    if lease['resource_key']=='venus-paced-recovery-v33':expected=(18000,600)
    assert (prior,duration)==expected
    assert lease['prior_gpu_seconds']==54451 and lease['prior_new_gpu_seconds_upper']==prior and lease['prior_scope_usd']==91.98448055555555
    cap_seconds=18600 if lease['resource_key'] in ('venus-paced-recovery-v32','venus-paced-recovery-v33') else 17300 if lease['resource_key'] in ('venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33') else 14400
    cap_usd=70 if lease['resource_key'] in ('venus-paced-recovery-v29','venus-paced-recovery-v30','venus-paced-recovery-v31','venus-paced-recovery-v32','venus-paced-recovery-v33') else 40
    assert 0<lease['verified_hourly_price_usd']<=4 and prior+duration<=cap_seconds and 54451+prior+duration<=(73051 if lease['resource_key'] in ('venus-paced-recovery-v32','venus-paced-recovery-v33') else 72000)
    assert (prior+duration)*4/3600+10+6+8<=cap_usd
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ['a2-','a3-','a4-','g2-']))]
    assert len(running)==1 and str(running[0]['id'])==r['ident'] and all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    return end
