"""Exact four-hour adaptive phase, carrying all previous time and cost."""
from pathlib import Path
import hashlib,json
from datetime import datetime

AUTH=Path('docs/NATIVE-VENUS-ADAPTIVE-PHASE-AUTHORIZATION-2026-10-29.json')
PLAN=Path('docs/NATIVE-VENUS-ADAPTIVE-PHASE-EXECUTION-2026-10-29.json')
KEY='venus-adaptive-v34'
RECOVERY_KEY='venus-adaptive-v35'
RETRY_KEY='venus-adaptive-v36'
EVAL_KEY='venus-adaptive-v37'
EVAL_PLAN=Path('docs/NATIVE-VENUS-ADAPTIVE-EVALUATION-RECOVERY-2026-10-29.json')
RETRY_PLAN=Path('docs/NATIVE-VENUS-ADAPTIVE-CONTROLLER-RECOVERY-2026-10-29.json')
RECOVERY_PLAN=Path('docs/NATIVE-VENUS-ADAPTIVE-RECOVERY-EXECUTION-2026-10-29.json')

def execution_plan(lease):
    if lease.get('resource_key')==EVAL_KEY:return EVAL_PLAN
    if lease.get('resource_key')==RETRY_KEY:return RETRY_PLAN
    return RECOVERY_PLAN if lease.get('resource_key')==RECOVERY_KEY else PLAN

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def resource(lease):
    plan=execution_plan(lease);a=json.loads(AUTH.read_text());p=json.loads(plan.read_text());r=a['resources']
    assert a['status']=='APPROVED' and a['human_approval']=='Approve four hours; keep $70 cap'
    assert a['parent_proposal_sha256']==sha(Path(a['parent_proposal']))
    assert (a['maximum_next_gpu_seconds'],a['maximum_additional_gcp_usage_usd'],a['maximum_candidate_updates'])==(14400,70,54)
    assert p['parent_sha256']==sha(AUTH) and lease['authorization_sha256']==sha(AUTH) and lease['execution_plan_sha256']==sha(plan)
    assert lease['resource_key'] in (KEY,RECOVERY_KEY,RETRY_KEY,EVAL_KEY) and lease['zone']==r['zone'] and lease['disk_id']==r['same_existing_disk_id']=='4375908109499109167'
    assert lease['disk_size_gib']==100 and str(lease['vm_id']).isdigit()
    return dict(name=p['vm_name'],zone=r['zone'],ident=str(lease['vm_id']),disk=r['same_existing_disk_id'],disk_name=r['same_existing_disk_name'],machine='a2-highgpu-1g',authority=AUTH)

def validate(vm,instances,pods,lease,now):
    r=resource(lease);p=json.loads(execution_plan(lease).read_text());duration=p['maximum_gpu_seconds'];start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+duration
    assert str(vm['id'])==r['ident'] and vm['name']==r['name'] and vm['status']=='RUNNING'
    assert vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone']) and vm['creationTimestamp']==lease['creation_timestamp']
    assert set(vm['tags']['items'])=={'duplex-venus-train-bounded'} and not vm.get('serviceAccounts')
    assert len(vm['disks'])==1 and any(d.get('boot') and not d.get('autoDelete') and d['source'].endswith('/disks/'+r['disk_name']) for d in vm['disks'])
    s=vm['scheduling'];assert s['provisioningModel']=='FLEX_START' and int(s['maxRunDuration']['seconds'])==duration
    assert s['instanceTerminationAction']=='STOP' and not s['automaticRestart'] and start<=now<end and lease['stop_unix']==end
    prior=p['focused_gpu_seconds_upper_before_launch']
    assert lease['prior_new_gpu_seconds_upper']==prior and prior>=18400 and 0<duration<=14400 and prior+duration==32800
    if lease['resource_key'] in (RECOVERY_KEY,RETRY_KEY,EVAL_KEY):
        last=p['previous_provider_snapshot'];assert last['status']=='TERMINATED' and start>=datetime.fromisoformat(last['lastStopTimestamp']).timestamp()
        elapsed=datetime.fromisoformat(last['lastStopTimestamp']).timestamp()-datetime.fromisoformat(last['lastStartTimestamp']).timestamp()
        if lease['resource_key']==RECOVERY_KEY:
            assert prior==18400+__import__('math').ceil(elapsed/100)*100
        else:
            intervals=p['closed_intervals']
            expected=['6583510368949016595','1283972191172335172']+(['6929308962022921268'] if lease['resource_key']==EVAL_KEY else [])
            assert [x['vm_id'] for x in intervals]==expected and intervals[-1]['vm_id']==last['id']
            assert abs(intervals[-1]['actual_seconds']-elapsed)<1e-6
            if lease['resource_key']==EVAL_KEY:
                assert p['new_optimizer_updates_authorized_by_this_execution']==0 and p['completed_candidate_updates']==54
            assert all(x['upper_seconds']==__import__('math').ceil(x['actual_seconds']/100)*100 for x in intervals)
            assert prior==18400+sum(x['upper_seconds'] for x in intervals)
    assert lease['prior_gpu_seconds']==54451 and lease['prior_scope_usd']==91.98448055555555
    assert 0<lease['verified_hourly_price_usd']<=4 and 54451+18400+14400==87251
    assert (18400+14400)*4/3600+10+6+8<=70 and 91.98448055555555+(18400+14400)*4/3600+10+6+8<=175
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ['a2-','a3-','a4-','g2-']))]
    assert len(running)==1 and str(running[0]['id'])==r['ident'] and all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    return end
