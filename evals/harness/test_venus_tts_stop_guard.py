import json
from pathlib import Path
from datetime import datetime
import pytest
from training.venus import tts_stop_guard as g


def test_original_resource_zero_updates_and_carried_budget(tmp_path,monkeypatch):
    from portable.venus import guard
    from training.venus.paced_guard import execution_plan
    vm=json.loads(Path('runs/venus-training-20261101/source-audit-v47/provider-start.json').read_text())
    assert vm['id']=='3873014408897233554'
    vm['scheduling']['maxRunDuration']['seconds']='1800'
    plan=dict(resource_key=g.KEY,authorization_sha256=g.sha(g.AUTH),new_optimizer_updates_authorized_by_this_execution=0,required_native_cases=12,maximum_gpu_seconds=1800,maximum_new_running_intervals=1)
    path=tmp_path/'execution.json';path.write_text(json.dumps(plan));monkeypatch.setattr(g,'PLAN',path)
    start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
    lease=dict(resource_key=g.KEY,authorization_sha256=g.sha(g.AUTH),execution_plan_sha256=g.sha(path),vm_id=vm['id'],disk_id='4375908109499109167',disk_size_gib=100,zone='us-central1-a',creation_timestamp=vm['creationTimestamp'],stop_unix=start+1800,prior_new_gpu_seconds_upper=33100,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=3.687)
    assert guard.resource(lease)['disk']==lease['disk_id'] and execution_plan(lease)==path
    assert guard.validate(vm,[vm],[],lease,start+1)==start+1800
    for change in (dict(prior_new_gpu_seconds_upper=0),dict(stop_unix=start+1900),dict(disk_id='123'),dict(resource_key='venus-context-train-v52')):
        with pytest.raises(AssertionError):g.validate(vm,[vm],[],dict(lease,**change),start+1)
    with pytest.raises(AssertionError):g.validate(vm,[vm,dict(vm,id='999')],[],lease,start+1)
    with pytest.raises(AssertionError):g.validate(vm,[vm],[dict(status='RUNNING')],lease,start+1)
    plan['new_optimizer_updates_authorized_by_this_execution']=60;path.write_text(json.dumps(plan));lease['execution_plan_sha256']=g.sha(path)
    with pytest.raises(AssertionError):g.resource(lease)
