import hashlib
import json
from datetime import datetime
from pathlib import Path

import pytest

from training.venus.action_guard import AUTH,PLAN,KEY,RETRY_KEY,execution_plan,validate,sha
from evals.harness.test_venus_adaptive_guard import fixture as old_fixture


def fixture(key=KEY):
    vm,lease,_=old_fixture()
    plan=execution_plan(dict(resource_key=key));p=json.loads(plan.read_text())
    start=datetime.fromisoformat('2026-10-30T00:30:00+00:00').timestamp()
    vm.update(name=p['vm_name'],lastStartTimestamp='2026-10-30T00:30:00Z')
    vm['scheduling']['maxRunDuration']['seconds']=str(p['maximum_gpu_seconds'])
    lease.update(resource_key=key,stop_unix=start+p['maximum_gpu_seconds'],prior_new_gpu_seconds_upper=p['focused_gpu_seconds_upper_before_launch'],
                 authorization_sha256=sha(AUTH),execution_plan_sha256=sha(plan))
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


@pytest.mark.parametrize('size',[100,200])
@pytest.mark.parametrize('key',[KEY,RETRY_KEY])
def test_actual_mac_guard_entrypoint_accepts_exact_authority_and_checks_disk(monkeypatch,tmp_path,size,key):
    from types import SimpleNamespace
    from portable.venus import guard
    vm,lease,start=fixture(key)
    lease.update(controller_pid=1,controller_birth='fixture-birth')
    path=tmp_path/'lease.json';path.write_text(json.dumps(lease))
    ssh=tmp_path/'ssh.json';ssh.write_text(json.dumps(['ssh','-o','StrictHostKeyChecking=yes','fixture']))
    key=tmp_path/'fixture-key';key.write_text('fixture-only-not-a-credential')
    stop=tmp_path/'stop';published=[]
    monkeypatch.setattr(guard,'run',lambda *a,**kw:'fixture-birth')
    monkeypatch.setattr(guard,'provider_snapshot',lambda *a:(vm,dict(id=lease['disk_id'],sizeGb=str(size)),[vm],[],start+1))
    monkeypatch.setattr(guard,'cloud',lambda *a,**kw:dict(vm,status='TERMINATED'))
    monkeypatch.setattr(guard.time,'time',lambda:start+1)
    monkeypatch.setattr(guard.time,'sleep',lambda _:None)
    def publish(*args):
        published.append(json.loads(args[1]));stop.write_text('fixture-complete')
    monkeypatch.setattr(guard,'publish_guard',publish)
    args=SimpleNamespace(lease=path,ssh_argv=ssh,runpod_key=key,status=tmp_path/'status',stop_file=stop)
    if size==100:
        guard.mac(args)
        assert len(published)==1 and published[0]['expires_unix']==start+91
    else:
        with pytest.raises(AssertionError):guard.mac(args)
        assert not published


def test_guard_recovery_carries_failed_startup_and_shortens_native_stop():
    vm,lease,start=fixture(RETRY_KEY)
    assert lease['prior_new_gpu_seconds_upper']==23400
    assert validate(vm,[vm],[],lease,start+1)==start+6900
    with pytest.raises(AssertionError):
        validate(vm,[vm],[],dict(lease,prior_new_gpu_seconds_upper=23100),start+1)
