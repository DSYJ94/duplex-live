"""Exact same-VM complete-epoch continuation within carried $70/time caps."""
import json
from pathlib import Path
from datetime import datetime
from .context_objective import ADMISSION,authority,sha

KEY='venus-context-train-v48'
PLAN=Path('docs/NATIVE-VENUS-CONTEXT-TRAINING-EXECUTION-2026-11-01.json')


def execution_plan(lease):
    assert lease['resource_key']==KEY
    return PLAN


def resource(lease):
    a=authority();p=json.loads(PLAN.read_text());r=a['resource']
    assert lease['resource_key']==KEY and lease['authorization_sha256']==sha(ADMISSION)
    assert p['admission_sha256']==sha(ADMISSION) and lease['execution_plan_sha256']==sha(PLAN)
    assert p['new_optimizer_updates_authorized_by_this_execution']==60 and p['maximum_gpu_seconds']==4100
    assert lease['zone']==r['zone'] and str(lease['vm_id'])==r['vm_id']
    assert lease['disk_id']==r['disk_id'] and lease['disk_size_gib']==100
    return dict(name=r['name'],zone=r['zone'],ident=r['vm_id'],disk=r['disk_id'],disk_name='duplex-venus-train-20261026-v1',machine=r['machine'],authority=ADMISSION,provider_read_limits=(25,20))


def validate(vm,instances,pods,lease,now):
    r=resource(lease);a=authority();p=json.loads(PLAN.read_text())
    start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+4000
    assert vm['name']==r['name'] and str(vm['id'])==r['ident'] and vm['status']=='RUNNING'
    assert vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone'])
    assert vm['creationTimestamp']==lease['creation_timestamp']
    assert set(vm['tags']['items'])=={'duplex-venus-train-bounded'} and not vm.get('serviceAccounts')
    assert len(vm['disks'])==1 and vm['disks'][0]['boot'] and not vm['disks'][0]['autoDelete']
    assert int(vm['disks'][0]['diskSizeGb'])==100 and vm['disks'][0]['source'].endswith('/disks/'+r['disk_name'])
    s=vm['scheduling']
    assert s['provisioningModel']=='STANDARD' and int(s['maxRunDuration']['seconds'])==4000
    assert s['instanceTerminationAction']=='STOP' and not s['automaticRestart']
    assert start<=now<end and lease['stop_unix']==end
    assert lease['prior_new_gpu_seconds_upper']==28700 and lease['prior_gpu_seconds']==54451 and lease['prior_scope_usd']==91.98448055555555
    assert 0<lease['verified_hourly_price_usd']<=4
    assert 28700+4100<=32800 and 54451+32800<=87251
    assert (28700+4100)*4/3600+33<=70
    assert 91.98448055555555+(28700+4100)*4/3600+33<=175
    assert start>=datetime.fromisoformat(p['previous_provider_stop']).timestamp()
    running=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ('a2-','a3-','a4-','g2-')))]
    assert len(running)==1 and str(running[0]['id'])==r['ident']
    assert all(x['status'] in ('EXITED','STOPPED','TERMINATED') for x in pods)
    return end
