import json
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

import pytest
from training.venus import context_placement_guard as cg
from training.venus.context_objective import authority,ADMISSION_V2,PLACEMENT_AUTH,sha
from training.venus.context_placement_controller import stockout_before_start


def fixture(tmp_path,monkeypatch):
    vm=json.loads(Path('runs/venus-training-20261101/source-audit-v47/provider-start.json').read_text())
    name='duplex-venus-context-20261102-c-v50';disk='duplex-venus-context-data-20261102-c-v50'
    vm.update(name=name,id='123',lastStartTimestamp='2026-11-02T10:00:00Z')
    vm['zone']=vm['zone'].replace('us-central1-a','us-central1-c')
    vm['disks'][0]['source']=vm['disks'][0]['source'].replace('us-central1-a','us-central1-c').rsplit('/',1)[0]+'/'+disk
    vm['scheduling']['maxRunDuration']['seconds']='3900'
    start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
    plan=dict(authorization_sha256=sha(PLACEMENT_AUTH),new_optimizer_updates_authorized_by_this_execution=60,maximum_gpu_seconds=3900,original_disk_id='4375908109499109167',snapshot_source_disk_id='4375908109499109167',zone='us-central1-c',vm_name=name,disk_name=disk,snapshot_name='duplex-venus-context-migrate-20261102-v50',vm_id='123',disk_id='456',snapshot_id='789',maximum_new_gpu_running_intervals=1)
    path=tmp_path/'execution.json';path.write_text(json.dumps(plan));monkeypatch.setattr(cg,'PLAN',path)
    lease=dict(resource_key=cg.KEY,authorization_sha256=sha(PLACEMENT_AUTH),execution_plan_sha256=sha(path),zone='us-central1-c',vm_id='123',disk_id='456',snapshot_id='789',disk_size_gib=100,creation_timestamp=vm['creationTimestamp'],stop_unix=start+3900,prior_new_gpu_seconds_upper=28700,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=3.7,storage_deadline_unix=start+8000)
    return vm,lease,start


def test_v2_binds_exact_approval_and_fixed_v3_without_replacing_v1():
    a=authority(ADMISSION_V2)
    assert a['maximum_new_updates']==60
    assert a['version']==2 and a['new_timeline_path'].endswith('context-timelines-v3')


def test_one_replacement_with_carried_cost_time_and_exact_clone(tmp_path,monkeypatch):
    vm,lease,start=fixture(tmp_path,monkeypatch)
    assert cg.validate(vm,[vm],[],lease,start+1)==start+3900
    for change in (dict(prior_new_gpu_seconds_upper=0),dict(disk_id='4375908109499109167'),dict(snapshot_id='999'),dict(stop_unix=start+4000),dict(storage_deadline_unix=start+3000)):
        with pytest.raises(AssertionError):cg.validate(vm,[vm],[],dict(lease,**change),start+1)
    with pytest.raises(AssertionError):cg.validate(vm,[vm,dict(vm,id='987')],[],lease,start+1)
    with pytest.raises(AssertionError):cg.validate(vm,[vm],[dict(status='RUNNING')],lease,start+1)


@pytest.mark.parametrize('actual_snapshot',['789','999'])
def test_actual_mac_entrypoint_verifies_restored_disk_ancestry(tmp_path,monkeypatch,actual_snapshot):
    from portable.venus import guard
    vm,lease,start=fixture(tmp_path,monkeypatch)
    lease.update(controller_pid=1,controller_birth='fixture-birth')
    path=tmp_path/'lease.json';path.write_text(json.dumps(lease))
    ssh=tmp_path/'ssh.json';ssh.write_text(json.dumps(['ssh','-o','StrictHostKeyChecking=yes','fixture']))
    key=tmp_path/'key';key.write_text('fixture-not-a-credential');stop=tmp_path/'stop';published=[]
    monkeypatch.setattr(guard,'run',lambda *a,**kw:'fixture-birth')
    monkeypatch.setattr(guard,'provider_snapshot',lambda *a:(vm,dict(id='456',sizeGb='100',sourceSnapshotId=actual_snapshot),[vm],[],start+1))
    monkeypatch.setattr(guard,'cloud',lambda *a,**kw:dict(vm,status='TERMINATED'))
    monkeypatch.setattr(guard.time,'time',lambda:start+1);monkeypatch.setattr(guard.time,'sleep',lambda _:None)
    def publish(*a):published.append(a);stop.write_text('done')
    monkeypatch.setattr(guard,'publish_guard',publish)
    args=SimpleNamespace(lease=path,ssh_argv=ssh,runpod_key=key,status=tmp_path/'status',stop_file=stop)
    if actual_snapshot=='789':guard.mac(args);assert len(published)==1
    else:
        with pytest.raises(AssertionError):guard.mac(args)
        assert not published


def test_only_actual_stockout_without_any_gpu_start_allows_next_zone():
    e=dict(error=dict(errors=[dict(code='ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS')]))
    assert stockout_before_start(e,None)
    assert stockout_before_start(e,dict(status='TERMINATED'))
    assert not stockout_before_start(e,dict(status='TERMINATED',lastStartTimestamp='2026-11-02T10:00:00Z'))
    assert not stockout_before_start(dict(error='PERMISSION_DENIED'),None)
