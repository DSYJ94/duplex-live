import ast,hashlib,inspect,sys,typing
from pathlib import Path
from types import SimpleNamespace,MethodType
from contextlib import nullcontext
import pytest
import torch
import torch.nn.functional as F
from portable.venus import tts_observer as observer

FIXTURE=Path('evals/fixtures/venus_tts_sampler_4681004.py')


def sampler(eos_at):
    node=next(x for x in ast.walk(ast.parse(FIXTURE.read_text())) if isinstance(x,ast.FunctionDef))
    node.body[0].value.value=inspect.cleandoc(node.body[0].value.value)
    assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()=='3d9aa1ce133e7079d0f8a5dc199b2562255b16e78d478ae953554f33d40a526f'
    ns=dict(torch=torch,F=F,Union=typing.Union,P=SimpleNamespace(cached=nullcontext),gen_logits=lambda **kw:([],[]))
    exec(compile(FIXTURE.read_text(),str(FIXTURE),'exec'),ns)
    state=SimpleNamespace(index=-1);cache=object()
    def model(**kwargs):
        state.index+=1
        return SimpleNamespace(last_hidden_state=torch.zeros(1,1,2),past_key_values=cache)
    def head(hidden):
        logits=torch.full((1,1,8),-10000.)
        logits[...,7 if state.index==eos_at else 3]=10000.
        return logits
    tts=SimpleNamespace(config=SimpleNamespace(num_audio_tokens=8),device='cpu',num_vq=1,num_audio_tokens=8,
        model=model,head_code=[head],emb_code=[lambda ids:torch.zeros(1,1,2)])
    tts.generate_chunk=MethodType(ns['generate_chunk'],tts)
    return tts,cache


def generate(tts):
    return tts.generate_chunk(inputs_embeds=torch.zeros(1,2,2),temperature=torch.ones(1),
        repetition_penalty=1.,eos_token=torch.tensor(7),max_new_token=26,min_new_tokens=0,
        force_no_stop=False,past_key_values=None,text_start_pos=0)


@pytest.mark.parametrize('eos_at,reason',[(25,'eos'),(999,'budget')])
def test_exact_upstream_sampler_same25_count_distinct_stop_and_unchanged_rng(monkeypatch,eos_at,reason):
    monkeypatch.setattr(observer,'SOURCE_SHA256',hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
    baseline,cache=sampler(eos_at);torch.manual_seed(42)
    expected,expected_cache=generate(baseline);rng=torch.get_rng_state()
    actual,cache=sampler(eos_at);original=actual.generate_chunk;torch.manual_seed(42)
    with observer.observe(actual) as records:
        tokens,returned_cache=generate(actual)
    assert expected.shape[1]==tokens.shape[1]==25
    assert torch.equal(tokens,expected) and torch.equal(torch.get_rng_state(),rng)
    assert returned_cache is cache and expected_cache is not None
    assert actual.generate_chunk is original and sys.getprofile() is None
    assert records[0]['termination']==reason and records[0]['sampled_steps']==26
    assert records[0]['discarded_non_eos_last_sample']==(reason=='budget')


def test_refuses_other_source_without_installing(monkeypatch):
    tts,_=sampler(2);original=tts.generate_chunk
    with pytest.raises(ValueError,match='pinned'):
        with observer.observe(tts):pass
    assert tts.generate_chunk is original


def test_profiler_and_method_restored_on_generation_error(monkeypatch):
    monkeypatch.setattr(observer,'SOURCE_SHA256',hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
    tts,_=sampler(2);original=tts.generate_chunk
    tts.model=lambda **kw:(_ for _ in ()).throw(ValueError('fixture-generation-error'))
    with pytest.raises(ValueError,match='fixture-generation-error'):
        with observer.observe(tts):generate(tts)
    assert sys.getprofile() is None and tts.generate_chunk is original


def test_conditioning_identity_and_converter_are_preserved(monkeypatch):
    monkeypatch.setattr(observer,'SOURCE_SHA256',hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
    tts,_=sampler(2);value=torch.zeros(1,3,2)
    convert=lambda rows:value
    duplex=SimpleNamespace(model=SimpleNamespace(tts=tts),_convert_results_to_tts_input=convert)
    with observer.observe_duplex(duplex) as records:
        assert duplex._convert_results_to_tts_input([[4,None,False],[5,None,True]]) is value
        generate(tts)
    assert records['conditioning']==[dict(native_token_ids=[4,5],native_rows=2,condition_tokens=3)]
    assert records['sampler'][0]['termination']=='eos'
    assert duplex._convert_results_to_tts_input is convert
