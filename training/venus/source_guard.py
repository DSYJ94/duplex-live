"""Source-only GCP qualification within the carried native training allowance."""
import hashlib,json
from pathlib import Path
from datetime import datetime
AUTH=Path('docs/NATIVE-VENUS-GCP-SOURCE-AUDIT-AUTHORIZATION-2026-11-01.json')
PLAN=Path('docs/NATIVE-VENUS-GCP-SOURCE-AUDIT-EXECUTION-2026-11-01.json')
KEY='venus-source-audit-v45'
RETRY_KEY='venus-source-audit-v46'
RETRY_PLAN=Path('docs/NATIVE-VENUS-GCP-SOURCE-AUDIT-RETRY-2026-11-01.json')
def plan_for(lease):
    assert lease['resource_key'] in (KEY,RETRY_KEY)
    return RETRY_PLAN if lease['resource_key']==RETRY_KEY else PLAN
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resource(lease):
    a=json.loads(AUTH.read_text());p=json.loads(plan_for(lease).read_text())
    assert a['status']=='APPROVED' and a['human_request']=='proceed with GCP for training'
    assert a['unchanged_phase_cap_usd']==70 and a['unchanged_cumulative_cap_usd']==175
    assert p['authorization_sha256']==sha(AUTH)==lease['authorization_sha256']
    assert lease['execution_plan_sha256']==sha(plan_for(lease)) and p['new_optimizer_updates']==0
    assert lease['resource_key'] in (KEY,RETRY_KEY) and lease['zone']=='us-central1-a'
    assert lease['disk_id']=='4375908109499109167' and lease['disk_size_gib']==100
    assert str(lease['vm_id']).isdigit()
    return dict(name=p['vm_name'],zone=lease['zone'],ident=str(lease['vm_id']),disk=lease['disk_id'],disk_name='duplex-venus-train-20261026-v1',machine='a2-highgpu-1g',authority=AUTH,provider_read_limits=(25,20))
def validate(vm,instances,pods,lease,now):
    r=resource(lease);p=json.loads(plan_for(lease).read_text())
    start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+1200
    assert vm['name']==r['name'] and str(vm['id'])==r['ident'] and vm['status']=='RUNNING'
    assert vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone'])
    assert vm['creationTimestamp']==lease['creation_timestamp']
    assert set(vm['tags']['items'])=={'duplex-venus-train-bounded'} and not vm.get('serviceAccounts')
    assert len(vm['disks'])==1 and vm['disks'][0]['boot'] and not vm['disks'][0]['autoDelete']
    assert int(vm['disks'][0]['diskSizeGb'])==100 and vm['disks'][0]['source'].endswith('/disks/'+r['disk_name'])
    s=vm['scheduling']
    assert s['provisioningModel']==p.get('provisioning_model','FLEX_START') and int(s['maxRunDuration']['seconds'])==1200
    assert s['instanceTerminationAction']=='STOP' and s['automaticRestart'] is False
    assert start<=now<end and lease['stop_unix']==end
    assert p['maximum_gpu_seconds']==1200 and p['focused_gpu_seconds_upper_before_launch']==lease['prior_new_gpu_seconds_upper']==28000
    assert lease['prior_gpu_seconds']==54451 and lease['prior_scope_usd']==91.98448055555555
    assert 0<lease['verified_hourly_price_usd']<=4
    assert 28000+1200<=32800 and 54451+28000+1200<=87251
    # Original24USD reserve PLUS both new5USD backend/source reservations.
    assert (28000+1200)*4/3600+34<=70
    assert 91.98448055555555+(28000+1200)*4/3600+34<=175
    assert start>=datetime.fromisoformat(p['previous_provider_stop']).timestamp()
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ('a2-','a3-','a4-','g2-')))]
    assert len(running)==1 and str(running[0]['id'])==r['ident']
    assert all(x['status'] in ('EXITED','STOPPED','TERMINATED') for x in pods)
    return end
