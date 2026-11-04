"""One proposed queued Flex-start comparison; approval required, zero updates."""
import json,hashlib
from pathlib import Path
from datetime import datetime
AUTH=Path('docs/NATIVE-VENUS-TTS-COMPLETION-FLEX-AUTHORIZATION-2026-11-04.json')
PLAN=Path('runs/venus-training-20261102/tts-completion-v62/execution.json')
KEY='venus-tts-completion-v62'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def execution_plan(lease):
    assert lease['resource_key']==KEY
    return PLAN


def resource(lease):
    a=json.loads(AUTH.read_text());p=json.loads(PLAN.read_text())
    assert lease['resource_key']==p['resource_key']==KEY
    assert a['status']=='APPROVED' and a['human_approval']=='Approve queued Flex-start within existing caps'
    assert sha(a['proposal_path'])==a['proposal_sha256']
    amendment=json.loads(Path(a['proposal_path']).read_text())
    prior=Path('docs/NATIVE-VENUS-TTS-STOP-DIAGNOSTIC-AUTHORIZATION-2026-11-02.json')
    assert sha(prior)==amendment['prior_diagnostic_authority_sha256']
    assert json.loads(prior.read_text())['status']=='APPROVED'
    assert sha(amendment['inherited_comparison'])==amendment['inherited_comparison_sha256']
    assert sha(amendment['failed_capacity_result'])==amendment['failed_capacity_result_sha256']
    failed=json.loads(Path(amendment['failed_capacity_result']).read_text())
    assert failed['actual_gpu_seconds']==failed['upper_gpu_seconds']==failed['new_model_cases']==failed['new_optimizer_updates']==0
    assert p['maximum_queue_seconds']==amendment['compute']['queue_wait_seconds_max']==3600
    assert lease['authorization_sha256']==p['authorization_sha256']==sha(AUTH)
    assert lease['execution_plan_sha256']==sha(PLAN) and p['new_optimizer_updates_authorized_by_this_execution']==0 and p['required_native_cases']==24
    assert p['maximum_gpu_seconds']==1200 and p['maximum_new_running_intervals']==1
    assert amendment['storage']['temporary_window_seconds']==7200
    assert p['storage_started_unix']==lease['storage_started_unix']
    assert p['storage_deadline_unix']==lease['storage_deadline_unix']==lease['storage_started_unix']+7200
    assert datetime.fromisoformat(a['recorded_utc']).timestamp()<=lease['storage_started_unix']<=datetime.fromisoformat(lease['creation_timestamp']).timestamp()
    assert lease['zone']=='us-central1-a'
    assert lease['vm_id']==p['vm_id'] and lease['disk_id']==p['disk_id'] and lease['snapshot_id']==p['snapshot_id']
    assert p['snapshot_id']==amendment['storage']['reuse_owned_snapshot_id']
    assert lease['disk_id']!='4375908109499109167' and lease['disk_size_gib']==100
    assert p['snapshot_source_disk_id']=='4375908109499109167'
    assert all(str(lease[k]).isdigit() for k in ('vm_id','disk_id','snapshot_id'))
    assert p['vm_name']==amendment['compute']['VM_name']=='duplex-venus-tts-flex-20261104-a-v62'
    assert p['disk_name']==amendment['storage']['disk_name']=='duplex-venus-tts-flex-data-20261104-a-v62'
    return dict(name=p['vm_name'],ident=lease['vm_id'],disk=lease['disk_id'],zone=lease['zone'],disk_name=p['disk_name'],machine='a2-highgpu-1g',authority=AUTH,provider_read_limits=(25,20))



def validate(vm,instances,pods,lease,now):
    r=resource(lease);start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+1200
    assert vm['id']==r['ident'] and vm['status']=='RUNNING' and vm['name']==r['name']
    assert vm['creationTimestamp']==lease['creation_timestamp'] and vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone'])
    assert set(vm['tags']['items'])=={'duplex-venus-train-bounded'} and not vm.get('serviceAccounts')
    assert len(vm['disks'])==1 and vm['disks'][0]['boot'] and not vm['disks'][0]['autoDelete'] and int(vm['disks'][0]['diskSizeGb'])==100
    assert vm['disks'][0]['source'].endswith('/disks/'+r['disk_name'])
    s=vm['scheduling'];assert s['provisioningModel']=='FLEX_START' and int(s['maxRunDuration']['seconds'])==1200 and s['instanceTerminationAction']=='STOP' and not s['automaticRestart']
    assert lease['stop_unix']==end and start<=now<end
    assert lease['prior_new_gpu_seconds_upper']==33500 and lease['prior_gpu_seconds']==54451 and lease['prior_scope_usd']==91.98448055555555
    assert 0<lease['verified_hourly_price_usd']<=4
    assert 33500+1300==34800 and 54451+34800==89251
    assert 34800*4/3600+33+2+.25<=75 and 91.98448055555555+34800*4/3600+33+2+.25<=175
    assert now<lease['storage_deadline_unix'] and end<lease['storage_deadline_unix']
    active=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ('a2-','a3-','a4-','g2-','g4-')))]
    assert len(active)==1 and active[0]['id']==r['ident']
    assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in pods)
    return end
