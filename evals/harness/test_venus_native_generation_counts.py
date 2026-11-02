"""Exercise the actual frame export without native weights or inference."""
import numpy as np
from evals.harness.test_venus_stage_timing import engine


def test_real_frame_export_retains_native_token_counts(monkeypatch):
    e=engine(monkeypatch,False,np.zeros(30720,dtype=np.float32))
    original=e.duplex.streaming_generate
    e.duplex.streaming_generate=lambda **kw:dict(original(**kw),n_tokens=10,n_tts_tokens=25)
    r=e.step(bytes(32000),None)
    assert r['native_generation_counts']==dict(text_tokens=10,audio_tokens=25)


def test_missing_or_invalid_native_counts_remain_unknown(monkeypatch):
    e=engine(monkeypatch,True,np.zeros(24000,dtype=np.float32))
    original=e.duplex.streaming_generate
    e.duplex.streaming_generate=lambda **kw:dict(original(**kw),n_tokens=True,n_tts_tokens=-1)
    r=e.step(bytes(32000),None)
    assert r['native_generation_counts']==dict(text_tokens=None,audio_tokens=None)
