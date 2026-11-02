"""Tiny Qwen gradient equivalence; not native9B or behavioral qualification."""
import pytest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from training.venus.prefix_cache import capture_prefix
from training.venus.lora import install,scene_loss
from evals.harness.test_venus_training import example


def test_four_layer_suffix_matches_full_native_forward_and_every_adapter_gradient():
    torch.manual_seed(42)
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=5,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,attention_dropout=0.)
    config._attn_implementation='eager';m=Qwen3ForCausalLM(config).eval();m.requires_grad_(False)
    e=example(m);e['segments']=[dict(start=i,length=1,return_logits=True) for i in range(6)]
    e['forward_path']='native-segmented-bptt-v1'
    prefix,proof=capture_prefix(m,e,frozen_layers=1)
    assert proof['passed'] and proof['logit_error']==proof['hidden_error']==0
    spec=install(m,rank=2,alpha=2,last_layers=4)
    assert len(spec['modules'])==8
    with torch.no_grad():
        for n,p in m.named_parameters():
            if p.requires_grad and n.endswith('.b'):p.normal_(std=.02)
    full,_=scene_loss(m,e,'cpu');full.backward()
    expected={n:p.grad.clone() for n,p in m.named_parameters() if p.requires_grad}
    assert len(expected)==16 and all(torch.count_nonzero(v)>0 for v in expected.values())
    m.zero_grad();same,_=scene_loss(m,prefix,'cpu');same.backward()
    assert torch.equal(full,same)
    assert all(torch.allclose(p.grad,expected[n],atol=1e-7,rtol=1e-5) for n,p in m.named_parameters() if p.requires_grad)
    assert all(p.grad is None for p in m.model.layers[0].parameters())
    assert prefix['prefix_layers']==1 and e['forward_path']=='native-segmented-bptt-v1'


@pytest.mark.parametrize('boundary',[-1,5,True,1.5])
def test_invalid_prefix_boundary_rejected_before_capture(boundary):
    from evals.harness.test_venus_training import model
    with pytest.raises(ValueError,match='boundary'):capture_prefix(model(),{},frozen_layers=boundary)
