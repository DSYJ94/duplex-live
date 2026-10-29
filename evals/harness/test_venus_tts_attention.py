from types import SimpleNamespace
import torch
import pytest
from transformers import LlamaConfig,LlamaModel
from portable.venus.tts_attention import configure_tts_attention


def model():
    torch.manual_seed(42)
    cfg=LlamaConfig(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0)
    cfg._attn_implementation='eager'
    tts=LlamaModel(cfg).eval().requires_grad_(False)
    return SimpleNamespace(tts=SimpleNamespace(model=tts),llm=SimpleNamespace(config=SimpleNamespace(_attn_implementation='sdpa')))


def test_tts_sdpa_prefill_and_cached_decode_preserve_tiny_cpu_outputs_and_weights():
    m=model();tokens=torch.tensor([[1,2,3]]);nexttoken=torch.tensor([[4]])
    before={n:p.clone() for n,p in m.tts.model.named_parameters()}
    with torch.no_grad():
        eager=m.tts.model(tokens,use_cache=True)
        eager_next=m.tts.model(nexttoken,past_key_values=eager.past_key_values,use_cache=True).last_hidden_state
    report=configure_tts_attention(m,'sdpa')
    with torch.no_grad():
        sdpa=m.tts.model(tokens,use_cache=True)
        sdpa_next=m.tts.model(nexttoken,past_key_values=sdpa.past_key_values,use_cache=True).last_hidden_state
    torch.testing.assert_close(eager.last_hidden_state,sdpa.last_hidden_state,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(eager_next,sdpa_next,rtol=1e-5,atol=1e-6)
    assert all(torch.equal(p,before[n]) for n,p in m.tts.model.named_parameters())
    assert m.llm.config._attn_implementation=='sdpa' and report['weights_unchanged']
    assert not report['native_audio_qualified']


def test_released_default_stays_eager_and_invalid_or_training_modes_fail():
    m=model();assert configure_tts_attention(m)['actual']=='eager'
    with pytest.raises(ValueError):configure_tts_attention(m,'made-up')
    m.tts.model.train()
    with pytest.raises(ValueError,match='frozen eval'):configure_tts_attention(m,'sdpa')
