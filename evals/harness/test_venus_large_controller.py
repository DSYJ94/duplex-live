import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_guest_resource_guard_has_only_standard_library_dependencies():
    result=subprocess.run([sys.executable,'-c',
        "import sys; import training.venus.large_guard; assert 'torch' not in sys.modules"],
        capture_output=True,text=True,check=True)
    assert result.returncode==0


def test_existing_gpu_claim_rejects_before_provider(monkeypatch,tmp_path):
    from training.venus import large_controller as c
    monkeypatch.setattr(c,'R',tmp_path)
    monkeypatch.setattr(c.c,'cloud',lambda *args:pytest.fail('Duplicate allocation'))
    with pytest.raises(ValueError,match='never duplicate'):c.main()


def test_copyback_over_100GB_stops_before_transfer(monkeypatch,tmp_path):
    from training.venus import large_controller as c
    monkeypatch.setattr(c,'R',tmp_path)
    monkeypatch.setattr(c,'BACKUP',tmp_path/'backup')
    monkeypatch.setattr(c.c,'ssh',lambda *args:json.dumps(dict(files=[dict(path='large',bytes=100_000_000_001,sha256='unused')],links=[])))
    monkeypatch.setattr(c.c,'run',lambda *args:pytest.fail('Unauthorized transfer'))
    with pytest.raises(ValueError,match='100GB'):c.copyback()
    assert not c.BACKUP.exists()


def test_pair_combination_rejects_passing_subset_or_changed_user_audio(tmp_path):
    from training.venus.large_behavior import combine
    from types import SimpleNamespace
    for arm in ('main','control'):
        folder=tmp_path/'native'/arm;folder.mkdir(parents=True)
        (folder/'native-report.json').write_text(json.dumps(dict(complete=True,cases=[dict(id=str(i)) for i in range(32)],source='source')))
        (folder/'results.json').write_text(json.dumps(dict(cases=[dict(id=str(i),condition=arm,input_sha256=('changed' if arm=='control' and i==0 else str(i))) for i in range(32)])))
    with pytest.raises(ValueError,match='hashes must match'):combine(SimpleNamespace(root=tmp_path,source='source'))
    assert not (tmp_path/'native/paired-native-report.json').exists()


def test_capacity_only_retry_requires_real_terminal_stockout_and_unchanged_provider(tmp_path,monkeypatch):
    from training.venus import large_controller as c
    monkeypatch.setattr(c,'ROOT',tmp_path)
    old=tmp_path/'gpu-05';old.mkdir()
    vm=dict(id='7521667153519853425',status='TERMINATED',lastStartTimestamp='2026-11-04T20:10:42+00:00',lastStopTimestamp='2026-11-04T20:17:45+00:00')
    (old/'no-new-running-interval.json').write_text(json.dumps(vm))
    op=dict(status='DONE',error=dict(errors=[dict(code='ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS')]))
    (old/'allocation-terminal.json').write_text(json.dumps(op))
    (old/'events.json').write_text(json.dumps([dict(event='one_value_selected_GPU_request',at_utc='2026-11-04T20:54:15+00:00')]))
    from datetime import datetime
    assert c.capacity_retry_after(1,vm)==datetime.fromisoformat('2026-11-04T21:04:15+00:00').timestamp()
    with pytest.raises(ValueError):c.capacity_retry_after(3,vm)
    with pytest.raises(ValueError):c.capacity_retry_after(1,dict(vm,lastStartTimestamp='changed'))
    op['error']['errors'][0]['code']='PERMISSION_DENIED'
    (old/'allocation-terminal.json').write_text(json.dumps(op))
    with pytest.raises(ValueError):c.capacity_retry_after(1,vm)
