"""Exact four-hour adaptive phase, carrying all previous time and cost."""
from pathlib import Path
import hashlib,json
from datetime import datetime

AUTH=Path('docs/NATIVE-VENUS-ADAPTIVE-PHASE-AUTHORIZATION-2026-10-29.json')
PLAN=Path('docs/NATIVE-VENUS-ADAPTIVE-PHASE-EXECUTION-2026-10-29.json')
KEY='venus-adaptive-v34'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def resource(lease):
    a=json.loads(AUTH.read_text());p=json.loads(PLAN.read_text());r=a['resources']
    assert a['status']=='APPROVED' and a['human_approval']=='Approve four hours; keep $70 cap'
    assert a['parent_proposal_sha256']==sha(Path(a['parent_proposal']))
    assert (a['maximum_next_gpu_seconds'],a['maximum_additional_gcp_usage_usd'],a['maximum_candidate_updates'])==(14400,70,54)
    assert p['parent_sha256']==sha(AUTH) and lease['authorization_sha256']==sha(AUTH) and lease['execution_plan_sha256']==sha(PLAN)
    assert lease['resource_key']==KEY and lease['zone']==r['zone'] and lease['disk_id']==r['same_existing_disk_id']=='4375908109499109167'
    assert lease['disk_size_gib']==100 and str(lease['vm_id']).isdigit()
    return dict(name=p['vm_name'],zone=r['zone'],ident=str(lease['vm_id']),disk=r['same_existing_disk_id'],disk_name=r['same_existing_disk_name'],machine='a2-highgpu-1g',authority=AUTH)

def validate(vm,instances,pods,lease,now):
    r=resource(lease);p=json.loads(PLAN.read_text());start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+14400
    assert str(vm['id'])==r['ident'] and vm['name']==r['name'] and vm['status']=='RUNNING'
    assert vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone']) and vm['creationTimestamp']==lease['creation_timestamp']
    assert set(vm['tags']['items'])=={'duplex-venus-train-bounded'} and not vm.get('serviceAccounts')
    assert len(vm['disks'])==1 and any(d.get('boot') and not d.get('autoDelete') and d['source'].endswith('/disks/'+r['disk_name']) for d in vm['disks'])
    s=vm['scheduling'];assert s['provisioningModel']=='FLEX_START' and int(s['maxRunDuration']['seconds'])==14400
    assert s['instanceTerminationAction']=='STOP' and not s['automaticRestart'] and start<=now<end and lease['stop_unix']==end
    assert lease['prior_new_gpu_seconds_upper']==p['focused_gpu_seconds_upper_before_launch']==18400
    assert lease['prior_gpu_seconds']==54451 and lease['prior_scope_usd']==91.98448055555555
    assert 0<lease['verified_hourly_price_usd']<=4 and 54451+18400+14400==87251
    assert (18400+14400)*4/3600+10+6+8<=70 and 91.98448055555555+(18400+14400)*4/3600+10+6+8<=175
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ['a2-','a3-','a4-','g2-']))]
    assert len(running)==1 and str(running[0]['id'])==r['ident'] and all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    return end
