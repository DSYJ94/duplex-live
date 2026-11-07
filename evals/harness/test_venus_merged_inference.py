import torch
from torch import nn
import pytest
from training.venus.lora import LoRALinear
from training.venus.merged_inference import merge_for_inference


def model(dtype=torch.float32):
    m=nn.Sequential(LoRALinear(nn.Linear(5,4,bias=True,dtype=dtype),rank=2,alpha=3))
    with torch.no_grad():
        m[0].b.copy_(torch.randn_like(m[0].b)*.2)
        m[0].extra_a=nn.Parameter(torch.randn(1,5)*.1)
        m[0].extra_b=nn.Parameter(torch.randn(4,1)*.1)
        m[0].extra_scale=.7
    return m.eval().requires_grad_(False)


def test_folding_includes_both_branches_bias_and_preserves_original_tensors_and_rng():
    torch.manual_seed(4);m=model();original=m[0]
    snapshots={n:p.detach().clone() for n,p in original.named_parameters()}
    x=torch.randn(3,5);before=m(x);rng=torch.get_rng_state().clone()
    receipt=merge_for_inference(m)
    torch.testing.assert_close(m(x),before,rtol=2e-6,atol=2e-6)
    assert torch.equal(torch.get_rng_state(),rng)
    assert type(m[0]) is nn.Linear and not any(p.requires_grad for p in m.parameters())
    assert all(torch.equal(p,snapshots[n]) for n,p in original.named_parameters())
    assert not receipt['bit_equivalence_claimed'] and not receipt['native_behavior_qualified']


def test_bfloat16_folding_does_not_claim_bit_equivalence():
    m=model(torch.bfloat16);original=m[0]
    expected=(original.base.weight.float()+(original.b@original.a)*1.5+(original.extra_b@original.extra_a)*.7).bfloat16()
    merge_for_inference(m)
    assert torch.equal(m[0].weight,expected)


@pytest.mark.parametrize('cause', ['training','gradients','disabled','nonfinite'])
def test_trainable_disabled_and_nonfinite_models_rejected_before_mutation(cause):
    m=model();original=m[0]
    if cause=='training':m.train()
    if cause=='gradients':m[0].a.requires_grad_(True)
    if cause=='disabled':m[0].enabled=False
    if cause=='nonfinite':
        with torch.no_grad():m[0].b[0,0]=float('nan')
    with pytest.raises(ValueError):merge_for_inference(m)
    assert m[0] is original


def test_late_cast_overflow_rolls_back_all_earlier_replacements():
    m=nn.Sequential(model(torch.float16)[0],model(torch.float16)[0]).eval().requires_grad_(False)
    originals=list(m)
    with torch.no_grad():m[1].a.fill_(1e20);m[1].b.fill_(1e20)
    with pytest.raises(ValueError):merge_for_inference(m)
    assert all(m[i] is p for i,p in enumerate(originals))
