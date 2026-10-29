import base64
from types import SimpleNamespace
import numpy as np
import torch
from portable.venus import native_app
from portable.venus.timing import stage_timings


def engine(monkeypatch,listen,audio):
    monkeypatch.setattr(native_app,'ensure_guard',lambda:None)
    e=native_app.NativeEngine.__new__(native_app.NativeEngine);e.np=np;e.torch=torch;e.cursor=0
    e.profile={k:1 for k in ('decode_mode','max_new_speak_tokens_per_chunk','temperature','top_k','top_p','listen_prob_scale')}
    result=dict(text='',audio_waveform=audio,is_listen=listen,end_of_turn=False,
        cost_llm=.2,cost_tts_prep=.1,cost_tts=.7,cost_token2wav=.3,cost_all=1.3)
    e.duplex=SimpleNamespace(streaming_prefill=lambda **kwargs:{'success':True},
        streaming_generate=lambda **kwargs:result,total_ids=[],tokenizer=SimpleNamespace(decode=lambda *args,**kwargs:''))
    return e


def test_native_timings_do_not_invent_zero_for_missing_or_invalid_stages():
    result=stage_timings({'cost_llm':.3,'cost_tts':float('nan')},prefill_seconds=.1,generation_seconds=.8)
    assert result['native_reported_wall_seconds']['cost_llm']==.3
    assert result['native_reported_wall_seconds']['cost_tts'] is None
    assert result['native_reported_wall_seconds']['cost_token2wav'] is None
    assert result['invalid_native_timing_fields']==['cost_tts']
    assert not result['gpu_kernel_profiler']


def test_listening_zero_waveform_is_not_sent_over_tunnel(monkeypatch):
    r=engine(monkeypatch,True,np.zeros(24000,dtype=np.float32)).step(bytes(32000),None)
    assert r['audio']=='' and r['audio_samples']==24000 and r['peak']==0
    assert r['wire_silence_omitted'] and r['wire_audio_bytes']==0
    assert r['timings']['native_reported_wall_seconds']['cost_tts']==.7


def test_speech_and_unexpected_nonzero_listen_audio_are_preserved_exactly(monkeypatch):
    wave=np.linspace(-.4,.4,24000,dtype=np.float32)
    for listen in (False,True):
        r=engine(monkeypatch,listen,wave).step(bytes(32000),None)
        assert base64.b64decode(r['audio'])==wave.astype('<f4').tobytes()
        assert not r['wire_silence_omitted'] and r['wire_audio_bytes']==96000
        assert r['timings']['prefill_wall_seconds']>=0


def test_stream_benchmark_missing_speaking_frames_are_unknown_not_zero_latency():
    from evals.venus_stream_benchmark import summary
    result=summary([dict(is_listen=True,processing_seconds=.4)])
    assert result['listen']['processing_mean']==.4
    assert result['speak']['frames']==0 and result['speak']['processing_mean'] is None
