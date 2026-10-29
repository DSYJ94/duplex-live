import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from training.venus.action_guard import AUTH,PLAN,KEY,validate,sha
from evals.harness.test_venus_adaptive_guard import fixture as old_fixture


def fixture():
    vm,lease,_=old_fixture()
    p=json.loads(PLAN.read_text())
    start=datetime.fromisoformat('2026-10-29T23:30:00+00:00').timestamp()
    vm.update(name=p['vm_name'],lastStartTimestamp='2026-10-29T23:30:00Z')
    vm['scheduling']['maxRunDuration']['seconds']='7200'
    lease.update(resource_key=KEY,stop_unix=start+7200,prior_new_gpu_seconds_upper=23100,
                 authorization_sha256=sha(AUTH),execution_plan_sha256=sha(PLAN))
    return vm,lease,start


def test_action_phase_keeps_all_prior_usage_and_only_one_gpu():
    vm,lease,start=fixture()
    assert validate(vm,[vm],[],lease,start+1)==start+7200
    with pytest.raises(AssertionError):validate(vm,[vm,dict(vm,id='other')],[],lease,start+1)
    with pytest.raises(AssertionError):validate(vm,[vm],[dict(status='RUNNING')],lease,start+1)


@pytest.mark.parametrize('change',[dict(prior_new_gpu_seconds_upper=0),dict(prior_gpu_seconds=0),
    dict(disk_id='wrong'),dict(stop_unix=0),dict(verified_hourly_price_usd=5)])
def test_action_phase_rejects_reset_budget_disk_deadline_or_price(change):
    vm,lease,start=fixture()
    with pytest.raises(AssertionError):validate(vm,[vm],[],dict(lease,**change),start+1)


def test_admitted_actual_complete_index_is_usable_without_changing_targets():
    from training.venus.action_objective import ActionObjective
    index=Path('runs/venus-training-20261029/action-target-audit-v1/index-verified-v2.json')
    cache=Path('runs/venus-training-20261029/terminal-render-v40/copyback/paced-prefix-cache-v24/manifest.json')
    if not index.exists():pytest.skip('Preserved local native corpus is optional in CI')
    value=ActionObjective(index,cache)
    assert value.weight==.25 and len(value.rows)==268


def test_new_parent_requires_explicit298_admission():
    from training.venus.native_optimizer import verified_parent
    state=dict(step=54,metadata=dict(parent_training_steps=244,source='dd767d65ac663e701142f2c882aeb5e1c2a2c6e5'),
               optimizer=dict(opal=1,update=54,version=1))
    digest='e7340edd15537523db78120a97902399faa0af9caa46e875ec92e56889d6edef'
    assert verified_parent(state,digest,allow298=True)==298
    with pytest.raises(ValueError):verified_parent(state,digest,allow244=True)
