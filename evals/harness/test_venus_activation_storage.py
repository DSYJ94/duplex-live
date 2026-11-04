"""Lossless storage counterfactual; does not qualify native GPU admission."""
import pytest
import torch
from training.venus.offload import ActivationOffload


def policy(model=None, **kw):
    return ActivationOffload(model or torch.nn.Linear(2,2), minimum_bytes=1,
                             force_cpu_test=True, policy='repeat4-pageable-v1', **kw)


def same_bytes(a,b):
    return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def test_native_shaped_repeated_kv_is_exact_and_quarter_storage():
    original=torch.randn(1,8,1537,128,dtype=torch.bfloat16).repeat_interleave(4,dim=1)
    hook=policy();packed=hook.pack(original)
    assert packed[3]==4 and hook.saved_bytes*4==hook.logical_bytes
    assert hook.compressed_tensors==1 and not packed[1].is_pinned()
    assert same_bytes(original,hook.unpack(packed))


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_nonrepeated_and_signed_zero_are_never_equated(dtype):
    x=torch.ones(1,4,3,8,dtype=dtype);x[:,0,0,0]=0.;x[:,1,0,0]=-0.
    hook=policy();packed=hook.pack(x)
    assert packed[3]==1 and same_bytes(x,hook.unpack(packed))
    y=torch.randn(1,4,3,8,dtype=dtype).transpose(2,3)
    packed=hook.pack(y)
    assert packed[3]==1 and same_bytes(y,hook.unpack(packed))


def test_actual_host_headroom_and_physical_reservation_reject_before_copy():
    x=torch.ones(1,4,3,8)
    hook=policy(available_host_fn=lambda:16*1024**3)
    with pytest.raises(MemoryError,match='headroom'):hook.pack(x)
    assert hook.saved_bytes==0
    hook=policy(maximum_host_bytes=1)
    with pytest.raises(MemoryError,match='reservation'):hook.pack(x)
    assert hook.saved_bytes==0


def test_native_segmented_gqa_loss_and_all_gradients_are_exact():
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from training.venus.lora import install,scene_loss
    from evals.harness.test_venus_training import example
    torch.manual_seed(501)
    c=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
                  num_attention_heads=4,num_key_value_heads=1,head_dim=8,attention_dropout=0.)
    c._attn_implementation='sdpa';m=Qwen3ForCausalLM(c).eval();install(m,2,2)
    with torch.no_grad():
        for n,p in m.named_parameters():
            if p.requires_grad and n.endswith('.b'):p.normal_(std=.02)
    e=example(m);e['segments']=[dict(start=i,length=1,return_logits=True) for i in range(6)]
    e['forward_path']='native-segmented-bptt-v1'
    loss,_=scene_loss(m,e,'cpu');loss.backward()
    expected={n:p.grad.clone() for n,p in m.named_parameters() if p.requires_grad}
    m.zero_grad();hook=policy(m)
    with hook.context():
        actual,_=scene_loss(m,e,'cpu');actual.backward()
    assert torch.equal(loss,actual) and hook.saved_bytes>0
    assert all(torch.equal(expected[n],p.grad) for n,p in m.named_parameters() if p.requires_grad)
