"""Lossless storage counterfactual; does not qualify native GPU admission."""
import pytest
import torch
from training.venus.offload import ActivationOffload


def policy(model=None, memory_policy='repeat4-pageable-v1', **kw):
    return ActivationOffload(model or torch.nn.Linear(2,2), minimum_bytes=1,
                             force_cpu_test=True, policy=memory_policy, **kw)


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


@pytest.mark.parametrize('memory_policy',['repeat4-pageable-v1','repeat4-prefix-pageable-v2'])
def test_native_segmented_gqa_loss_and_all_gradients_are_exact(memory_policy):
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
    m.zero_grad();hook=policy(m,memory_policy=memory_policy)
    with hook.context():
        actual,_=scene_loss(m,e,'cpu');actual.backward()
    assert torch.equal(loss,actual) and hook.saved_bytes>0
    assert all(torch.equal(expected[n],p.grad) for n,p in m.named_parameters() if p.requires_grad)
    if memory_policy=='repeat4-prefix-pageable-v2':assert hook.prefix_reuses>0


def test_growing_native_shaped_history_preserves_every_old_version_without_quadratic_storage():
    torch.manual_seed(73)
    original=torch.randn(1,8,1537,128,dtype=torch.bfloat16)
    hook=policy(memory_policy='repeat4-prefix-pageable-v2')
    lengths=[1,3,17,32,60,512,1025,1537]
    handles=[hook.pack(original[:,:,:n].repeat_interleave(4,dim=1)) for n in lengths]
    assert hook.prefix_reuses==len(lengths)-1 and len(hook.prefix_histories)==1
    assert hook.saved_bytes==1*8*2048*128*2
    for n,handle in zip(lengths,handles):
        assert same_bytes(hook.unpack(handle),original[:,:,:n].repeat_interleave(4,dim=1))


def test_forked_equal_first_token_never_overwrites_saved_history_and_owns_input():
    original=torch.ones(1,2,5,8,dtype=torch.bfloat16)
    changed=original.clone();changed[:,:,3,0]=-0.
    hook=policy(memory_policy='repeat4-prefix-pageable-v2')
    one=hook.pack(original.repeat_interleave(4,dim=1))
    two=hook.pack(changed.repeat_interleave(4,dim=1))
    assert len(next(iter(hook.prefix_histories.values())))==2
    original.add_(10);changed.add_(10)
    assert same_bytes(hook.unpack(one),torch.ones(1,8,5,8,dtype=torch.bfloat16))
    expected=torch.ones(1,2,5,8,dtype=torch.bfloat16);expected[:,:,3,0]=-0.
    assert same_bytes(hook.unpack(two),expected.repeat_interleave(4,dim=1))


def test_shared_prefix_physical_cap_cannot_be_bypassed_by_growth_or_fork():
    x=torch.ones(1,2,5,8,dtype=torch.bfloat16)
    hook=policy(memory_policy='repeat4-prefix-pageable-v2',maximum_host_bytes=256)
    original=hook.pack(x.repeat_interleave(4,dim=1))
    assert hook.saved_bytes==256
    # Reusing existing exact bytes consumes no new physical storage.
    hook.pack(x.repeat_interleave(4,dim=1));assert hook.saved_bytes==256
    fork=x.clone();fork[:,:,3,0]=2
    with pytest.raises(MemoryError,match='reservation'):hook.pack(fork.repeat_interleave(4,dim=1))
    grown=torch.ones(1,2,9,8,dtype=torch.bfloat16)
    with pytest.raises(MemoryError,match='reservation'):hook.pack(grown.repeat_interleave(4,dim=1))
    assert hook.saved_bytes==256 and same_bytes(hook.unpack(original),x.repeat_interleave(4,dim=1))
