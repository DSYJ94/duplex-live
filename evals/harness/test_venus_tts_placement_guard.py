import copy,json
from pathlib import Path
from datetime import datetime
import pytest
from training.venus import tts_placement_guard as g
from training.venus.tts_placement import stockout_before_start


def test_placement_lease_zero_update_and_budget(tmp_path,monkeypatch):
    from portable.venus import guard
    from training.venus.paced_guard import execution_plan
    vm=json.loads(Path('runs/venus-training-20261101/source-audit-v47/provider-start.json').read_text())
    vm.update(id='111',name='duplex-venus-tts-20261102-f-v56',zone=vm['zone'].rsplit('/',1)[0]+'/us-central1-f')
    vm['disks'][0]['source']=vm['disks'][0]['source'].rsplit('/',1)[0]+'/duplex-venus-tts-data-20261102-f-v56'
    vm['scheduling']['maxRunDuration']['seconds']='1800'
    plan=dict(resource_key=g.KEY,authorization_sha256=g.sha(g.AUTH),new_optimizer_updates_authorized_by_this_execution=0,required_native_cases=12,maximum_gpu_seconds=1800,maximum_new_running_intervals=1,vm_id='111',disk_id='222',snapshot_id='333',snapshot_source_disk_id='4375908109499109167',vm_name=vm['name'],disk_name='duplex-venus-tts-data-20261102-f-v56')
    path=tmp_path/'execution.json';path.write_text(json.dumps(plan));monkeypatch.setattr(g,'PLAN',path)
    start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
    lease=dict(resource_key=g.KEY,authorization_sha256=g.sha(g.AUTH),execution_plan_sha256=g.sha(path),vm_id=vm['id'],disk_id='222',snapshot_id='333',disk_size_gib=100,zone='us-central1-f',creation_timestamp=vm['creationTimestamp'],stop_unix=start+1800,prior_new_gpu_seconds_upper=33100,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=3.687,storage_deadline_unix=start+14400)
    assert guard.resource(lease)['disk']=='222' and execution_plan(lease)==path
    assert guard.validate(vm,[vm],[],lease,start+1)==start+1800
    for changes in (dict(prior_new_gpu_seconds_upper=0),dict(stop_unix=start+1900),dict(disk_id='4375908109499109167'),dict(snapshot_id='444'),dict(zone='us-central1-a'),dict(storage_deadline_unix=start+1799)):
        with pytest.raises(AssertionError):g.validate(vm,[vm],[],dict(lease,**changes),start+1)
    with pytest.raises(AssertionError):g.validate(vm,[vm,dict(vm,id='999')],[],lease,start+1)
    with pytest.raises(AssertionError):g.validate(vm,[vm],[dict(status='RUNNING')],lease,start+1)
    for key,value in [('new_optimizer_updates_authorized_by_this_execution',1),('required_native_cases',11),('maximum_gpu_seconds',1900),('maximum_new_running_intervals',2)]:
        bad=dict(plan,**{key:value});path.write_text(json.dumps(bad));lease['execution_plan_sha256']=g.sha(path)
        with pytest.raises(AssertionError):g.resource(lease)


def test_retry_only_actual_prestart_capacity_failure():
    op=dict(error=dict(errors=[dict(code='ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS')]))
    assert stockout_before_start(op,None)
    assert stockout_before_start(op,dict(status='TERMINATED'))
    assert not stockout_before_start(op,dict(status='TERMINATED',lastStartTimestamp='2026-11-03T00:00:00Z'))
    assert not stockout_before_start(dict(error=dict(code='PERMISSION_DENIED')),None)
    assert not stockout_before_start(dict(status='DONE'),None)
