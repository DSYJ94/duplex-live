import json
import pytest
from training.venus import context_continuation_guard as cg
from training.venus.context_objective import authority,ADMISSION_V3,CONTINUATION_AUTH,sha
from evals.harness.test_venus_context_placement import fixture as old_fixture


def fixture(tmp_path,monkeypatch):
    vm,lease,start=old_fixture(tmp_path,monkeypatch)
    path=tmp_path/'execution.json';p=json.loads(path.read_text())
    p.update(authorization_sha256=sha(CONTINUATION_AUTH),maximum_gpu_seconds=6600,
        vm_name=p['vm_name'].replace('v50','v52'),disk_name=p['disk_name'].replace('v50','v52'))
    path.write_text(json.dumps(p));monkeypatch.setattr(cg,'PLAN',path)
    vm['name']=p['vm_name'];vm['disks'][0]['source']=vm['disks'][0]['source'].replace('v50','v52')
    vm['scheduling']['maxRunDuration']['seconds']='6600'
    lease.update(resource_key=cg.KEY,authorization_sha256=sha(CONTINUATION_AUTH),
        execution_plan_sha256=sha(path),prior_new_gpu_seconds_upper=30000,stop_unix=start+6600)
    return vm,lease,start


def test_v3_keeps_recipe_and_exact14_while_carrying_actual_spend():
    a=authority(ADMISSION_V3)
    assert a['maximum_phase_usd']==75 and a['maximum_new_updates']==60
    assert a['partial14_witness_sha256']==sha('docs/NATIVE-VENUS-CONTEXT-PARTIAL14-REUSE-WITNESS-2026-11-02.json')
    assert a['focused_prior_gpu_seconds_upper']==30000 and a['recipe']['train']==240


def test_new_interval_cannot_reset_spend_extend_time_or_overlap(tmp_path,monkeypatch):
    from portable.venus import guard
    from training.venus.paced_guard import execution_plan
    vm,lease,start=fixture(tmp_path,monkeypatch)
    assert cg.validate(vm,[vm],[],lease,start+1)==start+6600
    assert guard.validate(vm,[vm],[],lease,start+1)==start+6600
    assert guard.resource(lease)['disk']=='456'
    assert execution_plan(lease)==cg.PLAN
    for change in (dict(prior_new_gpu_seconds_upper=28700),dict(stop_unix=start+6700),
                   dict(resource_key='venus-context-train-v50'),dict(disk_id='4375908109499109167')):
        with pytest.raises(AssertionError):cg.validate(vm,[vm],[],dict(lease,**change),start+1)
    with pytest.raises(AssertionError):cg.validate(vm,[vm,dict(vm,id='99')],[],lease,start+1)
    with pytest.raises(AssertionError):cg.validate(vm,[vm],[dict(status='RUNNING')],lease,start+1)


def test_actual_mac_guard_accepts_new_authority_and_publishes_only_exact_ancestry(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from portable.venus import guard
    vm,lease,start=fixture(tmp_path,monkeypatch)
    lease.update(controller_pid=1,controller_birth='fixture-birth')
    lp=tmp_path/'lease.json';lp.write_text(json.dumps(lease))
    ssh=tmp_path/'ssh.json';ssh.write_text(json.dumps(['ssh','-o','StrictHostKeyChecking=yes','fixture']))
    key=tmp_path/'key';key.write_text('fixture-key');stop=tmp_path/'stop';published=[]
    monkeypatch.setattr(guard,'run',lambda *a,**kw:'fixture-birth')
    monkeypatch.setattr(guard,'cloud',lambda *a,**kw:dict(vm,status='TERMINATED'))
    monkeypatch.setattr(guard,'provider_snapshot',lambda *a:(vm,dict(id='456',sizeGb='100',sourceSnapshotId='789'),[vm],[],start+1))
    monkeypatch.setattr(guard.time,'time',lambda:start+1)
    monkeypatch.setattr(guard.time,'sleep',lambda _:None)
    def publish(*a):published.append(a);stop.write_text('done')
    monkeypatch.setattr(guard,'publish_guard',publish)
    args=SimpleNamespace(lease=lp,ssh_argv=ssh,runpod_key=key,status=tmp_path/'status',stop_file=stop)
    guard.mac(args)
    assert len(published)==1
