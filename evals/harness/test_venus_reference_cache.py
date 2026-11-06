"""Test and falsify omitted-reference fallback using the pinned actual methods."""
from contextlib import nullcontext
from pathlib import Path
from types import MethodType,SimpleNamespace

import pytest

FIXTURE=Path('evals/fixtures/venus_token2wav_reference_1_0_6.py')


class ReachedAcousticModel(Exception):pass


class Tensor:
    device='scripted';dtype='scripted';shape=(1,3,4)
    def __mul__(self,other):return self


def tokenizer():
    calls=[]
    def stop_before_CUDA(*a,**kw):raise ReachedAcousticModel
    fake=SimpleNamespace(int32='scripted',ones=lambda *a,**kw:Tensor(),zeros=lambda *a,**kw:Tensor(),cat=lambda *a,**kw:Tensor(),tensor=stop_before_CUDA)
    ns=dict(torch=fake)
    exec(compile(FIXTURE.read_text(),str(FIXTURE),'exec'),ns)
    t=SimpleNamespace(cache=None,n_timesteps=10,flow=SimpleNamespace(setup_cache=lambda *a,**kw:{}))
    def prepare(path):calls.append(path);return tuple(Tensor() for _ in range(5))
    t._prepare_prompt=prepare
    for name in ('set_stream_cache','stream'):setattr(t,name,MethodType(ns[name],t))
    return t,calls


def test_missing_frame_reference_does_not_replace_prepared_custom_cache():
    t,calls=tokenizer();t.set_stream_cache('chosen-British-reference.wav');cached=t.cache
    with pytest.raises(ReachedAcousticModel):t.stream([1,2],prompt_wav=None)
    assert calls==['chosen-British-reference.wav'] and t.cache is cached


def test_counterfactual_cache_loss_is_distinguishable_from_real_intact_cache():
    t,calls=tokenizer();t.set_stream_cache('chosen-British-reference.wav');t.cache=None
    with pytest.raises(ReachedAcousticModel):t.stream([1,2],prompt_wav=None)
    assert calls==['chosen-British-reference.wav',None]


def test_engine_prepare_reinitializes_exact_selected_reference_after_session_reset(monkeypatch):
    from portable.venus.native_app import NativeEngine
    monkeypatch.setattr('portable.venus.native_app.ensure_guard',lambda:None)
    t,calls=tokenizer();events=[]
    e=NativeEngine.__new__(NativeEngine)
    e.model=SimpleNamespace(reset_session=lambda:events.append('reset'))
    e.torch=SimpleNamespace(inference_mode=nullcontext);e.action_channel=None
    e.profile=dict(system_prompt='Streaming Omni Conversation.')
    e.voice_reference=dict(path='chosen-British-reference.wav')
    def prepare(**kw):
        assert events[-1]=='reset'
        t.cache=None;t.set_stream_cache(kw['prompt_wav_path'])
    e.duplex=SimpleNamespace(prepare=prepare)
    e.prepare();assert calls==['chosen-British-reference.wav']
    with pytest.raises(ReachedAcousticModel):t.stream([1],prompt_wav=None)
    assert calls==['chosen-British-reference.wav']
