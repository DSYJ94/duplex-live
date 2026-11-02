import json
import pytest
from training.venus import cpu_recovery as r


def test_guard_stops_on_gpu_or_wrong_resource(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'R',tmp_path)
    (tmp_path/'lease.json').write_text(json.dumps(dict(vm_id='123',stop_unix=100,controller_pid=1,controller_birth='birth')))
    vm=dict(id='123',status='RUNNING',machineType='/e2-standard-2',guestAccelerators=[{'acceleratorCount':1}],disks=[dict(source='/disks/'+r.DISK)])
    stopped=[]
    monkeypatch.setattr(r.c,'cloud',lambda *a:vm)
    monkeypatch.setattr(r,'stop',lambda:stopped.append(True))
    with pytest.raises(AssertionError):r.guard()
    assert stopped==[True]


def test_guard_observes_terminal_without_start_or_stop(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'R',tmp_path)
    (tmp_path/'lease.json').write_text(json.dumps(dict(vm_id='123')))
    monkeypatch.setattr(r.c,'cloud',lambda *a:dict(status='TERMINATED'))
    monkeypatch.setattr(r,'stop',lambda:pytest.fail('Already terminal'))
    r.guard()
