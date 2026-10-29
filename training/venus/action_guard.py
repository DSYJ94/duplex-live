"""Exact approved108-update phase; all prior time/cost remains carried."""
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

AUTH = Path('docs/NATIVE-VENUS-ACTION-FIDELITY-AUTHORIZATION-2026-10-29.json')
PLAN = Path('docs/NATIVE-VENUS-ACTION-FIDELITY-EXECUTION-2026-10-29.json')
KEY = 'venus-action-fidelity-v41'
RETRY_KEY = 'venus-action-fidelity-v42'
RETRY_PLAN = Path('docs/NATIVE-VENUS-ACTION-GUARD-RECOVERY-EXECUTION-2026-10-29.json')


def execution_plan(lease):
    if lease['resource_key'] not in (KEY,RETRY_KEY):raise ValueError('Known action phase lease required')
    return RETRY_PLAN if lease['resource_key']==RETRY_KEY else PLAN


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource(lease):
    plan=execution_plan(lease)
    a, p = json.loads(AUTH.read_text()), json.loads(plan.read_text())
    assert a['status'] == 'APPROVED' and a['human_approval'] == 'Approve 108 updates within existing time/$70 cap'
    assert a['proposal_sha256'] == sha(a['proposal_path'])
    assert a['maximum_candidate_new_updates'] == p['new_optimizer_updates_authorized_by_this_execution'] == 108
    assert a['unchanged_phase_cap_usd']==70 and a['unchanged_cumulative_cap_usd']==175
    assert a['maximum_gpu_seconds_from_existing_allowance']==7200 and 0<p['maximum_gpu_seconds']<=7200
    assert a['external_model_api_and_synthesis_calls'] == 0
    assert a['same_disk_id'] == lease['disk_id'] == '4375908109499109167' and lease['disk_size_gib'] == 100
    assert lease['resource_key'] in (KEY,RETRY_KEY) and lease['zone'] == a['same_zone'] == 'us-central1-a'
    assert lease['authorization_sha256'] == sha(AUTH) == p['parent_sha256']
    assert lease['execution_plan_sha256'] == sha(plan) and str(lease['vm_id']).isdigit()
    return dict(name=p['vm_name'], zone=lease['zone'], ident=str(lease['vm_id']),
                disk=lease['disk_id'], disk_name=p['same_disk_name'], machine='a2-highgpu-1g',
                authority=AUTH, provider_read_limits=(25,20))


def validate(vm, instances, pods, lease, now):
    r = resource(lease)
    p = json.loads(execution_plan(lease).read_text())
    start = datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
    end = start + p['maximum_gpu_seconds']
    assert str(vm['id']) == r['ident'] and vm['name'] == r['name'] and vm['status'] == 'RUNNING'
    assert vm['machineType'].endswith('/'+r['machine']) and vm['zone'].endswith('/'+r['zone'])
    assert vm['creationTimestamp'] == lease['creation_timestamp']
    assert set(vm['tags']['items']) == {'duplex-venus-train-bounded'} and not vm.get('serviceAccounts')
    assert len(vm['disks']) == 1 and vm['disks'][0]['boot'] and not vm['disks'][0]['autoDelete']
    assert vm['disks'][0]['source'].endswith('/disks/'+r['disk_name'])
    s = vm['scheduling']
    duration=p['maximum_gpu_seconds']
    assert s['provisioningModel'] == 'FLEX_START' and int(s['maxRunDuration']['seconds']) == duration
    assert s['instanceTerminationAction'] == 'STOP' and not s['automaticRestart']
    assert start <= now < end and lease['stop_unix'] == end
    intervals = p['closed_intervals']
    assert [x['vm_id'] for x in intervals] == ['6583510368949016595','1283972191172335172',
        '6929308962022921268','5559941523972223154','2738075597351611996',
        '4626264501959715272','49337671038724022']+(['4574352937354946887'] if lease['resource_key']==RETRY_KEY else [])
    assert all(x['upper_seconds'] == math.ceil(x['actual_seconds']/100)*100 for x in intervals)
    prior = 18400 + sum(x['upper_seconds'] for x in intervals)
    assert lease['prior_new_gpu_seconds_upper'] == p['focused_gpu_seconds_upper_before_launch'] == prior
    assert prior>=23100 and prior+duration==23100+7200
    if lease['resource_key']==RETRY_KEY:
        assert p['previous_attempt_new_optimizer_updates']==p['previous_attempt_model_units_started']==0
    assert lease['prior_gpu_seconds'] == 54451 and lease['prior_scope_usd'] == 91.98448055555555
    assert 0 < lease['verified_hourly_price_usd'] <= 4
    assert prior+duration <= 32800 and 54451+prior+duration <= 87251
    assert (prior+duration)*4/3600+24 <= 70
    assert 91.98448055555555+(prior+duration)*4/3600+24 <= 175
    assert start >= datetime.fromisoformat(p['previous_provider_snapshot']['lastStopTimestamp']).timestamp()
    running = [v for v in instances if v['status'] not in ('TERMINATED','STOPPED','STOPPING')
               and (v.get('guestAccelerators') or any('/'+m in v.get('machineType','') for m in ('a2-','a3-','a4-','g2-')))]
    assert len(running) == 1 and str(running[0]['id']) == r['ident']
    assert all(p['status'] in ('EXITED','STOPPED','TERMINATED') for p in pods)
    return end
