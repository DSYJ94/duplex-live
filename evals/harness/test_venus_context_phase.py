import copy
import json
from datetime import datetime
from pathlib import Path
import unittest
from types import SimpleNamespace
import pytest

from training.venus.context_objective import authority,ADMISSION,sha
from training.venus.context_guard import validate,PLAN,KEY


class ContextPhaseTests(unittest.TestCase):
    def fixture(self):
        vm=json.loads(Path('runs/venus-training-20261101/source-audit-v47/provider-start.json').read_text())
        vm['scheduling']['maxRunDuration']['seconds']='4000'
        lease=json.loads(Path('runs/venus-training-20261101/source-audit-v47/lease.json').read_text())
        now=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
        vm['lastStartTimestamp']='2026-11-02T00:00:00Z'
        now=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
        lease.update(resource_key=KEY,authorization_sha256=sha(ADMISSION),execution_plan_sha256=sha(PLAN),prior_new_gpu_seconds_upper=28700,stop_unix=now+4000)
        return vm,lease,now

    def test_whole_epoch_and_cumulative_cost_admitted(self):
        a=authority()
        self.assertEqual((a['maximum_new_updates'],a['maximum_complete_epochs']),(60,1))
        vm,lease,now=self.fixture()
        self.assertEqual(validate(vm,[vm],[],lease,now),now+4000)

    def test_wrong_disk_deadline_overlap_or_budget_reset_rejected(self):
        for kind in ('disk','deadline','overlap','reset'):
            vm,lease,now=self.fixture();instances=[vm]
            if kind=='disk':lease['disk_id']='123'
            elif kind=='deadline':lease['stop_unix']+=1
            elif kind=='overlap':instances.append(dict(vm,id='123'))
            else:lease['prior_new_gpu_seconds_upper']=0
            with self.subTest(kind=kind),self.assertRaises(AssertionError):validate(vm,instances,[],lease,now)

    def test_no_running_runpod_or_different_provider_stop_policy(self):
        vm,lease,now=self.fixture()
        with self.assertRaises(AssertionError):validate(vm,[vm],[dict(status='RUNNING')],lease,now)
        vm['scheduling']['instanceTerminationAction']='DELETE'
        with self.assertRaises(AssertionError):validate(vm,[vm],[],lease,now)


@pytest.mark.parametrize('size',[100,200])
def test_actual_mac_entrypoint_checks_real_disk_size(monkeypatch,tmp_path,size):
    from portable.venus import guard
    vm,lease,start=ContextPhaseTests().fixture()
    lease.update(controller_pid=1,controller_birth='fixture-only-birth')
    path=tmp_path/'lease.json';path.write_text(json.dumps(lease))
    ssh=tmp_path/'ssh.json';ssh.write_text(json.dumps(['ssh','-o','StrictHostKeyChecking=yes','fixture']))
    key=tmp_path/'fixture-key';key.write_text('fixture-not-a-credential')
    stop=tmp_path/'stop';published=[]
    monkeypatch.setattr(guard,'run',lambda *a,**kw:'fixture-only-birth')
    monkeypatch.setattr(guard,'provider_snapshot',lambda *a:(vm,dict(id=lease['disk_id'],sizeGb=str(size)),[vm],[],start+1))
    monkeypatch.setattr(guard,'cloud',lambda *a,**kw:dict(vm,status='TERMINATED'))
    monkeypatch.setattr(guard.time,'time',lambda:start+1)
    monkeypatch.setattr(guard.time,'sleep',lambda _:None)
    def publish(*args):
        published.append(json.loads(args[1]));stop.write_text('complete')
    monkeypatch.setattr(guard,'publish_guard',publish)
    args=SimpleNamespace(lease=path,ssh_argv=ssh,runpod_key=key,status=tmp_path/'status',stop_file=stop)
    if size==100:
        guard.mac(args)
        assert len(published)==1
    else:
        with pytest.raises(AssertionError):guard.mac(args)
        assert not published


def test_capacity_retry_requires_actual_terminal_stockout_before_any_model():
    from training.venus.context_controller import retry_ready
    prior=Path('runs/venus-training-20261101/context-v48')
    vm=json.loads((prior/'provider-final.json').read_text())
    at=datetime.fromisoformat('2026-11-01T23:42:00Z').timestamp()
    retry_ready(prior,vm,at)
    with pytest.raises(ValueError):retry_ready(prior,vm,at-301)
    with pytest.raises(ValueError):retry_ready(prior,dict(vm,status='RUNNING'),at)
