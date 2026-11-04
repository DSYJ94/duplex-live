"""Exercise the real probe entrypoint without loading a model or doing ASR."""
import json,sys,time
from pathlib import Path
from types import SimpleNamespace
import pytest


def test_paired_nested_output_reaches_model_boundary_without_overwrite(tmp_path,monkeypatch):
    from training.venus import tts_stop_probe as probe
    from portable.venus import native_app
    # The entrypoint imports the codec package, but this test stops before any
    # codec/model operation. Inputs below are explicitly fixture bytes, not WAVs.
    monkeypatch.setitem(sys.modules,'soundfile',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(set_seed=lambda _:None))
    monkeypatch.setattr(native_app,'ensure_guard',lambda:None)
    root=tmp_path/'timelines';scenes=[]
    for i in range(6):
        folder=root/str(i);folder.mkdir(parents=True)
        (folder/'user.wav').write_bytes(b'fixture-only-no-audio-decoding')
        (folder/'timeline.json').write_text('{}')
        scenes.append(dict(id=str(i),family='ordinary',user_wav_sha256=probe.sha(folder/'user.wav'),timeline_sha256=probe.sha(folder/'timeline.json')))
    selection=tmp_path/'selection.json';selection.write_text(json.dumps(dict(scenes=scenes,required_native_cases=12,conditions=['untouched-release','candidate466'],terminal_budget_unchanged=26)))
    lease=tmp_path/'lease.json';lease.write_text(json.dumps(dict(stop_unix=time.time()+7200)))
    checkpoint=tmp_path/'fixture-checkpoint';checkpoint.write_text('fixture-only')
    original_sha=probe.sha
    monkeypatch.setattr(probe,'sha',lambda p:'e2c16cb825dc654519e922379a220b8c962e9d8a78eec652a3b195922a69720f' if Path(p)==checkpoint else original_sha(p))
    out=tmp_path/'not-yet-created-comparison'/'baseline'
    class ReachedModel(Exception):pass
    def model(*a,**kw):
        assert out.is_dir()
        raise ReachedModel
    monkeypatch.setattr(native_app,'NativeEngine',model)
    args=SimpleNamespace(model='not-loaded',checkpoint=str(checkpoint),selection=str(selection),timelines=str(root),voice_reference='not-read',voice_reference_sha256='fixture',lease=str(lease),output=str(out),minimum_start_seconds=600)
    with pytest.raises(ReachedModel):probe.main(args)
    receipt=out/'preserved-receipt';receipt.write_text('keep')
    with pytest.raises(FileExistsError):probe.main(args)
    assert receipt.read_text()=='keep'
