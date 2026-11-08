"""A complete new-condition audit preserves clipping and fails on missing inputs."""
import argparse,json,sys,types
import numpy as np
import pytest
from training.venus import foundation_native as native
from training.personaplex.foundation_context import VARIANTS


def test_raw_keys_and_shapes_are_complete_not_permissive():
    from training.personaplex.foundation_context import check_raw_shapes
    class Model:
        dep_q=16
        def state_dict(self):return {'head':np.zeros((4,3)), 'depth':np.zeros((16,2))}
    assert check_raw_shapes(Model(),{'head':(4,3),'depth':(16,2)})['native_depth_heads']==16
    for shapes in ({'head':(4,3)}, {'head':(4,3),'depth':(8,2)}, {'head':(4,3),'depth':(16,2),'extra':(1,)}):
        with pytest.raises(ValueError,match='Exact raw checkpoint'):check_raw_shapes(Model(),shapes)


def test_all32_moshi_outputs_keep_original_peak_and_no_worker(tmp_path,monkeypatch):
    sf=pytest.importorskip('soundfile');pytest.importorskip('scipy.signal')
    cases=[dict(id=str(i),family='ordinary',input_sha256=str(i),generated_text='hello') for i in range(32)]
    root=tmp_path/VARIANTS[1];root.mkdir();(root/'results.json').write_text(json.dumps(dict(cases=cases)))
    for c in cases:
        folder=root/c['id'];folder.mkdir();sf.write(folder/'chunk.wav',np.full(1920,1.2,dtype=np.float32),24000,subtype='FLOAT')
        (folder/'trace.jsonl').write_text(json.dumps(dict(sequence=2,chunk_file='chunk.wav'))+'\n')
    monkeypatch.setattr(native,'cases',lambda work:cases)
    monkeypatch.setattr(native,'guarded',lambda *a:None)
    monkeypatch.setitem(sys.modules,'training.venus.voice_compare',types.SimpleNamespace(recognizer=lambda:lambda *a,**kw:dict(text='hello')))
    result=native.audit_conditions(argparse.Namespace(root=str(tmp_path),source='source',lease='lease'),(VARIANTS[1],))
    assert result[VARIANTS[1]]['word_pass']==32 and result[VARIANTS[1]]['raw_peak_pass']==0
    report=json.loads((root/'report.json').read_text())
    assert not report['adopted'] and all(not c['native_worker_interface'] for c in report['cases'])
    timed,sr=sf.read(root/'0/timed-emitted-raw.wav',dtype='float32')
    assert sr==24000 and len(timed)==5760 and timed.max()>1
    np.testing.assert_array_equal(timed[:3840],np.zeros(3840))
    cases.pop();(root/'results.json').write_text(json.dumps(dict(cases=cases)))
    with pytest.raises(ValueError,match='All32 exact'):native.audit_conditions(argparse.Namespace(root=str(tmp_path),source='source',lease='lease'),(VARIANTS[1],))
