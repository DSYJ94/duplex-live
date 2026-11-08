"""Raw-audio evidence stays unclipped and matched inputs fail closed."""
import json
import numpy as np
import pytest
from training.venus import foundation_native as native


def test_timed_assembly_preserves_overlap_and_raw_clipping():
    a=np.array([.8,.8],dtype=np.float32)
    result=native.timed_assembly([dict(sequence=0),dict(sequence=1)],[a,a],1)
    np.testing.assert_allclose(result,[.8,1.6,.8])
    assert result.dtype==np.float32 and result.max()>1


def test_moshi_80ms_positions_retain_gap():
    result=native.timed_assembly([dict(sequence=0),dict(sequence=2)], [np.ones(2),np.ones(2)],100,frame_seconds=.08)
    np.testing.assert_array_equal(result[2:16],np.zeros(14))
    assert len(result)==18


def test_cohort_requires_original_hash_and_all32(tmp_path,monkeypatch):
    import sys,types
    fresh=[dict(id=str(i),family='quiet',frames=2,input_sha256=str(i)) for i in range(24)]
    old=[dict(id=str(i),family='ordinary',frames=3,input_sha256=str(i)) for i in range(24,32)]
    ref=[dict(id=r['id'],family=r['family'],input_frames=r['frames'],input_sha256=r['input_sha256']) for r in fresh+old]
    folder=tmp_path/'native-posttrain646-v1/native/main';folder.mkdir(parents=True)
    (folder/'results.json').write_text(json.dumps(dict(cases=ref)))
    docs=tmp_path/'docs';docs.mkdir()
    (docs/'NATIVE-RELEASE-CURRENT32-INPUTS-2026-11-07.json').write_text(json.dumps(dict(reference_collection_sha256=native.sha(folder/'results.json'))))
    monkeypatch.setattr(native,'BASE',tmp_path)
    monkeypatch.setitem(sys.modules,'training.venus.behavior',types.SimpleNamespace(preflight=lambda plan,*a: fresh if 'repair-source' in str(plan) else old))
    assert len(native.cases(tmp_path))==32
    old[0]['input_sha256']='changed'
    with pytest.raises(ValueError,match='input changed'):native.cases(tmp_path)
    old.pop()
    with pytest.raises(ValueError,match='Every original32'):native.cases(tmp_path)
