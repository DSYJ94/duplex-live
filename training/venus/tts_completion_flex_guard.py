"""One proposed queued Flex-start comparison; approval required, zero updates."""
import json,hashlib
from pathlib import Path
from datetime import datetime
AUTH=Path('docs/NATIVE-VENUS-TTS-COMPLETION-FLEX-AUTHORIZATION-2026-11-04.json')
PLAN=Path('runs/venus-training-20261102/tts-completion-v62/execution.json')
RECOVERY_AUTH=Path('docs/NATIVE-VENUS-TTS-COMPLETION-FLEX-RECOVERY-AUTHORIZATION-2026-11-04.json')
RECOVERY_PLAN=Path('runs/venus-training-20261102/tts-completion-v63/execution.json')
KEY='venus-tts-completion-v62'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding(lease):
    assert lease['resource_key']==KEY
    version=lease.get('execution_version',62)
    assert version in (62,63)
    return (RECOVERY_AUTH,RECOVERY_PLAN,1000,33700,63) if version==63 else (AUTH,PLAN,1200,33500,62)


def execution_plan(lease):
    return binding(lease)[1]


def read_scope(a,version):
    expected='Approve recovery within unused allowance' if version==63 else 'Approve queued Flex-start within existing caps'
    assert a['status']=='APPROVED' and a['human_approval']==expected
    assert sha(a['proposal_path'])==a['proposal_sha256']
    amendment=json.loads(Path(a['proposal_path']).read_text());base=amendment
    if version==63:
        assert sha(amendment['prior_authorization'])==amendment['prior_authorization_sha256']
        assert json.loads(Path(amendment['prior_authorization']).read_text())['status']=='APPROVED'
        assert sha(amendment['prior_proposal'])==amendment['prior_proposal_sha256']
        base=json.loads(Path(amendment['prior_proposal']).read_text())
        assert sha(amendment['startup_result'])==amendment['startup_result_sha256']
        failed=json.loads(Path(amendment['startup_result']).read_text())
        assert failed['upper_gpu_seconds']==200 and 0<failed['actual_gpu_seconds']<=200
        assert failed['model_constructions']==failed['native_cases_completed']==failed['optimizer_updates']==0
        assert amendment['budget']['current_prior_focused_seconds_upper']==33700
        assert amendment['budget']['remaining_new_GPU_seconds_upper']==1100
    return amendment,base


def resource(lease):
    authority,plan,duration,prior_seconds,version=binding(lease)
    a=json.loads(authority.read_text());p=json.loads(plan.read_text())
    assert lease['resource_key']==p['resource_key']==KEY
    assert a.get('execution_version',62)==p.get('execution_version',62)==version
    amendment,base=read_scope(a,version)
    prior=Path('docs/NATIVE-VENUS-TTS-STOP-DIAGNOSTIC-AUTHORIZATION-2026-11-02.json')
    assert sha(prior)==base['prior_diagnostic_authority_sha256']
    assert json.loads(prior.read_text())['status']=='APPROVED'
    assert sha(base['inherited_comparison'])==base['inherited_comparison_sha256']
    assert sha(base['failed_capacity_result'])==base['failed_capacity_result_sha256']
    failed=json.loads(Path(base['failed_capacity_result']).read_text())
    assert failed['actual_gpu_seconds']==failed['upper_gpu_seconds']==failed['new_model_cases']==failed['new_optimizer_updates']==0
    assert p['maximum_queue_seconds']==amendment['compute']['queue_wait_seconds_max']==3600
    assert lease['authorization_sha256']==p['authorization_sha256']==sha(authority)
    assert lease['execution_plan_sha256']==sha(plan) and p['new_optimizer_updates_authorized_by_this_execution']==0 and p['required_native_cases']==24
    assert p['maximum_gpu_seconds']==duration and p['maximum_new_running_intervals']==1
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
    assert p['vm_name']==amendment['compute']['VM_name']=='duplex-venus-tts-flex-20261104-a-v'+str(version)
    assert p['disk_name']==amendment['storage']['disk_name']=='duplex-venus-tts-flex-data-20261104-a-v'+str(version)
    return dict(name=p['vm_name'],ident=lease['vm_id'],disk=lease['disk_id'],zone=lease['zone'],disk_name=p['disk_name'],machine='a2-highgpu-1g',authority=authority,provider_read_limits=(25,20),duration=duration,prior_seconds=prior_seconds)



def validate(vm,instances,pods,lease,now):
    r=resource(lease);start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+r['duration']
    assert vm['id']==r['ident'] and vm['status']=='RUNNING' and vm['name']==r['name']
    assert vm['creationTimestamp']==lease['creation_timestamp'] and vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone'])
    assert set(vm['tags']['items'])=={'duplex-venus-train-bounded'} and not vm.get('serviceAccounts')
    assert len(vm['disks'])==1 and vm['disks'][0]['boot'] and not vm['disks'][0]['autoDelete'] and int(vm['disks'][0]['diskSizeGb'])==100
    assert vm['disks'][0]['source'].endswith('/disks/'+r['disk_name'])
    s=vm['scheduling'];assert s['provisioningModel']=='FLEX_START' and int(s['maxRunDuration']['seconds'])==r['duration'] and s['instanceTerminationAction']=='STOP' and not s['automaticRestart']
    assert lease['stop_unix']==end and start<=now<end
    assert lease['prior_new_gpu_seconds_upper']==r['prior_seconds'] and lease['prior_gpu_seconds']==54451 and lease['prior_scope_usd']==91.98448055555555
    assert 0<lease['verified_hourly_price_usd']<=4
    assert r['prior_seconds']+r['duration']+100==34800 and 54451+34800==89251
    assert 34800*4/3600+33+2+.25<=75 and 91.98448055555555+34800*4/3600+33+2+.25<=175
    assert now<lease['storage_deadline_unix'] and end<lease['storage_deadline_unix']
    active=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ('a2-','a3-','a4-','g2-','g4-')))]
    assert len(active)==1 and active[0]['id']==r['ident']
    assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in pods)
    return end
