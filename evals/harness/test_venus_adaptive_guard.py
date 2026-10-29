import copy,hashlib,json
from datetime import datetime
import pytest
from training.venus.adaptive_guard import AUTH,PLAN,KEY,validate

def fixture():
    p=json.loads(PLAN.read_text());start=datetime.fromisoformat('2026-10-29T11:00:00+00:00').timestamp()
    lease=dict(resource_key=KEY,zone='us-central1-a',vm_id='123',disk_id='4375908109499109167',disk_size_gib=100,creation_timestamp='2026-10-29T10:59:55Z',stop_unix=start+14400,prior_new_gpu_seconds_upper=18400,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=3.687083630136986,authorization_sha256=hashlib.sha256(AUTH.read_bytes()).hexdigest(),execution_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest())
    vm=dict(id='123',name=p['vm_name'],status='RUNNING',machineType='/a2-highgpu-1g',zone='/us-central1-a',creationTimestamp=lease['creation_timestamp'],lastStartTimestamp='2026-10-29T11:00:00Z',tags={'items':['duplex-venus-train-bounded']},disks=[dict(boot=True,autoDelete=False,source='/disks/duplex-venus-train-20261026-v1')],scheduling=dict(provisioningModel='FLEX_START',maxRunDuration={'seconds':'14400'},instanceTerminationAction='STOP',automaticRestart=False))
    return vm,lease,start

def test_exact_scope_and_carry_are_valid():
    vm,lease,start=fixture();assert validate(vm,[vm],[],lease,start+1)==start+14400

@pytest.mark.parametrize('change',[dict(prior_new_gpu_seconds_upper=0),dict(prior_gpu_seconds=0),dict(disk_id='other'),dict(stop_unix=99999999999),dict(verified_hourly_price_usd=5)])
def test_cannot_reset_usage_or_change_disk_deadline_or_rate(change):
    vm,lease,start=fixture()
    with pytest.raises(AssertionError):validate(vm,[vm],[],dict(lease,**change),start+1)

def test_second_gpu_or_paid_runpod_is_rejected():
    vm,lease,start=fixture()
    with pytest.raises(AssertionError):validate(vm,[vm,dict(vm,id='other')],[],lease,start+1)
    with pytest.raises(AssertionError):validate(vm,[vm],[{'status':'RUNNING'}],lease,start+1)


def test_recovery_carries_actual_closed_startup_and_only_uses_remaining_seconds():
    from training.venus.adaptive_guard import RECOVERY_KEY,RECOVERY_PLAN
    p=json.loads(RECOVERY_PLAN.read_text());vm,lease,_=fixture()
    start=datetime.fromisoformat('2026-10-29T12:00:00+00:00').timestamp()
    duration=p['maximum_gpu_seconds']
    vm.update(lastStartTimestamp='2026-10-29T12:00:00Z',creationTimestamp='2026-10-29T11:59:55Z')
    vm['scheduling']['maxRunDuration']['seconds']=str(duration)
    lease.update(resource_key=RECOVERY_KEY,creation_timestamp=vm['creationTimestamp'],stop_unix=start+duration,
                 prior_new_gpu_seconds_upper=p['focused_gpu_seconds_upper_before_launch'],execution_plan_sha256=hashlib.sha256(RECOVERY_PLAN.read_bytes()).hexdigest())
    assert validate(vm,[vm],[],lease,start+1)==start+duration
    with pytest.raises(AssertionError):validate(vm,[vm],[],dict(lease,prior_new_gpu_seconds_upper=18400),start+1)
