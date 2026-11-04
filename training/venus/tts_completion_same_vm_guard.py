"""Approved existing-VM complete24 speech comparison; zero optimizer updates."""
import json,hashlib
from pathlib import Path
from datetime import datetime
AUTH=Path('docs/NATIVE-VENUS-TTS-COMPLETION-SAME-VM-AUTHORIZATION-2026-11-04.json')
PLAN=Path('runs/venus-training-20261102/tts-completion-v60/execution.json')
RETRY_AUTH=Path('docs/NATIVE-VENUS-TTS-COMPLETION-SAME-VM-RETRY-AUTHORIZATION-2026-11-04.json')
RETRY_PLAN=Path('runs/venus-training-20261102/tts-completion-v61/execution.json')
KEY='venus-tts-completion-v60'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def binding(lease):
    assert lease['resource_key']==KEY
    version=lease.get('execution_version',60)
    assert version in (60,61),'Unknown execution must not borrow an existing grant'
    return (RETRY_AUTH,RETRY_PLAN) if version==61 else (AUTH,PLAN)


def execution_plan(lease):
    return binding(lease)[1]


def resource(lease):
    authority,plan=binding(lease)
    a=json.loads(authority.read_text());p=json.loads(plan.read_text())
    assert lease['resource_key']==p['resource_key']==KEY
    assert p.get('execution_version',60)==a.get('execution_version',60)==lease.get('execution_version',60)
    assert a['status']=='APPROVED' and a['human_approval']=='Approve existing-VM retry within existing caps'
    assert sha(a['proposal_path'])==a['proposal_sha256']
    assert lease['authorization_sha256']==p['authorization_sha256']==sha(authority)
    assert lease['execution_plan_sha256']==sha(plan) and p['new_optimizer_updates_authorized_by_this_execution']==0 and p['required_native_cases']==24
    if lease.get('execution_version',60)==61:
        assert sha(a['previous_authorization_path'])==a['previous_authorization_sha256']
        prior=json.loads(Path(a['previous_authorization_path']).read_text());assert prior['status']=='APPROVED'
        assert datetime.fromisoformat(prior['recorded_utc'])<=datetime.fromisoformat(a['recorded_utc'])
        assert sha(a['previous_execution_result'])==a['previous_execution_result_sha256']
        failed=json.loads(Path(a['previous_execution_result']).read_text())
        assert failed['actual_gpu_seconds']==failed['upper_gpu_seconds']==failed['new_optimizer_updates']==failed['new_model_cases']==0
    assert p['maximum_gpu_seconds']==1200 and p['maximum_new_running_intervals']==1
    assert lease['vm_id']=='3873014408897233554' and lease['disk_id']=='4375908109499109167' and lease['zone']=='us-central1-a'
    assert lease['disk_size_gib']==100
    proposal=json.loads(Path(a['proposal_path']).read_text())
    assert sha(proposal['failed_capacity_result'])==proposal['failed_capacity_result_sha256']
    assert sha(proposal['complete_evaluation_inherited_from'])==proposal['complete_evaluation_sha256']
    assert p['maximum_prestart_attempts']==3 and 1<=p['actual_start_attempt_index']<=3
    assert p['prestart_started_unix']==lease['prestart_started_unix']
    assert p['prestart_deadline_unix']==lease['prestart_deadline_unix']==lease['prestart_started_unix']+2700
    assert datetime.fromisoformat(a['recorded_utc']).timestamp()<=lease['prestart_started_unix']
    return dict(name='duplex-venus-source-20261101-v46',ident=lease['vm_id'],disk=lease['disk_id'],zone=lease['zone'],disk_name='duplex-venus-train-20261026-v1',machine='a2-highgpu-1g',authority=authority,provider_read_limits=(25,20))


def validate(vm,instances,pods,lease,now):
    r=resource(lease);start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();end=start+1200
    assert vm['id']==r['ident'] and vm['status']=='RUNNING' and vm['name']==r['name']
    assert vm['creationTimestamp']==lease['creation_timestamp'] and vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone'])
    assert set(vm['tags']['items'])=={'duplex-venus-train-bounded'} and not vm.get('serviceAccounts')
    assert len(vm['disks'])==1 and vm['disks'][0]['boot'] and not vm['disks'][0]['autoDelete'] and int(vm['disks'][0]['diskSizeGb'])==100
    assert vm['disks'][0]['source'].endswith('/disks/'+r['disk_name'])
    s=vm['scheduling'];assert s['provisioningModel']=='STANDARD' and int(s['maxRunDuration']['seconds'])==1200 and s['instanceTerminationAction']=='STOP' and not s['automaticRestart']
    assert lease['stop_unix']==end and start<=now<end
    assert lease['prestart_started_unix']<=start<=lease['prestart_deadline_unix']
    assert lease['prior_new_gpu_seconds_upper']==33500 and lease['prior_gpu_seconds']==54451 and lease['prior_scope_usd']==91.98448055555555
    assert 0<lease['verified_hourly_price_usd']<=4
    assert 33500+1300==34800 and 54451+34800==89251
    assert 34800*4/3600+33+2+.25<=75 and 91.98448055555555+34800*4/3600+33+2+.25<=175
    active=[v for v in instances if v['status'] not in ('TERMINATED','STOPPED') and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ('a2-','a3-','a4-','g2-','g4-')))]
    assert len(active)==1 and active[0]['id']==r['ident']
    assert all(v['status'] in ('EXITED','STOPPED','TERMINATED') for v in pods)
    return end
