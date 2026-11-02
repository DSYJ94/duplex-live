"""Exact approved original-VM diagnostic; zero optimizer updates."""
import json,hashlib
from pathlib import Path
from datetime import datetime
AUTH=Path('docs/NATIVE-VENUS-TTS-STOP-DIAGNOSTIC-AUTHORIZATION-2026-11-02.json')
PLAN=Path('runs/venus-training-20261102/tts-stop-v55/execution.json')
KEY='venus-tts-stop-v55'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def execution_plan(lease):
    assert lease['resource_key']==KEY
    return PLAN


def resource(lease):
    a=json.loads(AUTH.read_text());p=json.loads(PLAN.read_text())
    assert lease['resource_key']==p['resource_key']==KEY
    assert a['status']=='APPROVED' and a['human_approval']=='Approve 30 minutes within existing caps'
    assert sha(a['proposal_path'])==a['proposal_sha256']
    assert lease['authorization_sha256']==p['authorization_sha256']==sha(AUTH)
    assert lease['execution_plan_sha256']==sha(PLAN) and p['new_optimizer_updates_authorized_by_this_execution']==0 and p['required_native_cases']==12
    assert p['maximum_gpu_seconds']==1800 and p['maximum_new_running_intervals']==1
    assert lease['vm_id']=='3873014408897233554' and lease['disk_id']=='4375908109499109167' and lease['zone']=='us-central1-a'
    assert lease['disk_size_gib']==100
    return dict(name='duplex-venus-source-20261101-v46',ident=lease['vm_id'],disk=lease['disk_id'],zone=lease['zone'],disk_name='duplex-venus-train-20261026-v1',machine='a2-highgpu-1g',authority=AUTH,provider_read_limits=(25,20))


def validate(vm,instances,pods,lease,now):
    r=resource(lease);start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+1800
    assert vm['id']==r['ident'] and vm['status']=='RUNNING' and vm['name']==r['name']
    assert vm['creationTimestamp']==lease['creation_timestamp'] and vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone'])
    assert set(vm['tags']['items'])=={'duplex-venus-train-bounded'} and not vm.get('serviceAccounts')
    assert len(vm['disks'])==1 and vm['disks'][0]['boot'] and not vm['disks'][0]['autoDelete'] and int(vm['disks'][0]['diskSizeGb'])==100
    assert vm['disks'][0]['source'].endswith('/disks/'+r['disk_name'])
    s=vm['scheduling'];assert s['provisioningModel']=='STANDARD' and int(s['maxRunDuration']['seconds'])==1800 and s['instanceTerminationAction']=='STOP' and not s['automaticRestart']
    assert lease['stop_unix']==end and start<=now<end
    assert lease['prior_new_gpu_seconds_upper']==33100 and lease['prior_gpu_seconds']==54451 and lease['prior_scope_usd']==91.98448055555555
    assert 0<lease['verified_hourly_price_usd']<=4
    assert 33100+1900==35000 and 54451+35000==89451
    assert 35000*4/3600+33+1+.25<=75 and 91.98448055555555+35000*4/3600+33+1+.25<=175
    active=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ('a2-','a3-','a4-','g2-','g4-')))]
    assert len(active)==1 and active[0]['id']==r['ident']
    assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in pods)
    return end
