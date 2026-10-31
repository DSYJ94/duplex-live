"""Actual guard entrypoint coverage for the zero-update comparison lease."""
import json
from datetime import datetime
import pytest
from training.venus.runtime_guard import PLAN,AUTH,KEY,sha,validate
from evals.harness.test_venus_adaptive_guard import fixture as original


def fixture(key=KEY):
    vm,lease,_=original();p=json.loads(PLAN.read_text())
    start=datetime.fromisoformat('2026-10-31T14:00:00+00:00').timestamp()
    vm.update(name=p['vm_name'],lastStartTimestamp='2026-10-31T14:00:00Z')
    vm['scheduling']['maxRunDuration']['seconds']='2400'
    lease.update(resource_key=KEY,stop_unix=start+2400,prior_new_gpu_seconds_upper=26800,authorization_sha256=sha(AUTH),execution_plan_sha256=sha(PLAN))
    return vm,lease,start


@pytest.mark.parametrize('size',[100,200])
def test_actual_mac_entrypoint(monkeypatch,tmp_path,size):
    from evals.harness import test_venus_action_phase as prior
    monkeypatch.setattr(prior,'fixture',fixture)
    prior.test_actual_mac_guard_entrypoint_accepts_exact_authority_and_checks_disk(monkeypatch,tmp_path,size,KEY)


def test_budget_and_overlap():
    vm,lease,start=fixture()
    assert validate(vm,[vm],[],lease,start+1)==start+2400
    for change in [dict(prior_new_gpu_seconds_upper=0),dict(stop_unix=start+5000),dict(disk_id='other')]:
        with pytest.raises(AssertionError):validate(vm,[vm],[],dict(lease,**change),start+1)
    with pytest.raises(AssertionError):validate(vm,[vm,dict(vm,id='another')],[],lease,start+1)
    assert json.loads(PLAN.read_text())['new_optimizer_updates_authorized_by_this_execution']==0
