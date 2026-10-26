"""Small causal/gradient/restore checks; never presented as native qualification."""
import copy
import random
import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from training.venus.lora import install, disabled, selected_forward, scene_loss, adapter_state
from training.venus.state import save, load


def model():
    torch.manual_seed(42)
    c=Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                 num_attention_heads=4, num_key_value_heads=2, head_dim=8, attention_dropout=0.)
    c._attn_implementation='eager'
    return Qwen3ForCausalLM(c).eval()


def example(m):
    x=m.model.embed_tokens(torch.tensor([1,2,3,4,5,6])).detach()
    return dict(embeddings=x,positions=torch.tensor([2,3,4]),targets=torch.tensor([4,5,6]),
                decision_mask=torch.tensor([True,False,True]),family='ordinary')


def test_zero_adapter_parity_and_causal_mask():
    m=model(); e=example(m); before=selected_forward(m,e,'cpu')[0].detach(); spec=install(m,2,2)
    assert spec['parameters']==448
    after=selected_forward(m,e,'cpu')[0]
    assert torch.equal(before,after)
    changed=copy.deepcopy(e); changed['embeddings'][-1]+=10
    assert torch.equal(after,selected_forward(m,changed,'cpu')[0])
    bad=copy.deepcopy(e);bad['embeddings'][1]+=torch.arange(32)
    assert not torch.equal(after,selected_forward(m,bad,'cpu')[0])


def test_meaningful_gradients_and_frozen_weights():
    m=model()
    # The random tiny head otherwise bounds all possible logits close to zero;
    # scale its fixed readout so this is a meaningful learnability test.
    with torch.no_grad():m.lm_head.weight.mul_(4)
    e=example(m); install(m,4,4)
    frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.02)
    first=float(scene_loss(m,e,'cpu')[0]); losses=[]
    for _ in range(12):
        opt.zero_grad();loss,_=scene_loss(m,e,'cpu');assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
        assert any(p.grad.abs().max()>0 for p in m.parameters() if p.requires_grad)
        opt.step();losses.append(float(loss.detach()))
    assert losses[-1]<first*.8
    assert all(torch.equal(p,frozen[n]) for n,p in m.named_parameters() if n in frozen)


def test_exact_adam_rng_restore(tmp_path):
    random.seed(42);np.random.seed(42);m=model();e=example(m);install(m,2,2)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
    loss,_=scene_loss(m,e,'cpu');loss.backward();opt.step();opt.zero_grad()
    path=tmp_path/'state.pt';meta=dict(source='tiny-proof',data='exact',recipe=1)
    save(path,m,opt,1,meta)
    expected_rng=(random.random(),np.random.rand(),torch.rand(3))
    loss,_=scene_loss(m,e,'cpu');loss.backward();opt.step();expected=adapter_state(m)
    fresh=model();install(fresh,2,2);freshopt=torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad],lr=.01)
    assert load(path,fresh,freshopt,meta)==1
    actual_rng=(random.random(),np.random.rand(),torch.rand(3))
    assert expected_rng[:2]==actual_rng[:2] and torch.equal(expected_rng[2],actual_rng[2])
    freshopt.zero_grad();loss,_=scene_loss(fresh,e,'cpu');loss.backward();freshopt.step()
    assert all(torch.equal(v,adapter_state(fresh)[n]) for n,v in expected.items())
    with pytest.raises(ValueError):load(path,fresh,freshopt,dict(wrong=True))


def test_native_segment_to_training_alignment_is_red_capable():
    from types import SimpleNamespace
    from training.venus.native_cache import whole_sequence_proof
    m=model();e=example(m);e['positions']=torch.tensor([0,1,2,3,4]);e['targets']=torch.tensor([2,3,4,5,6])
    refs=[];cache=None
    with torch.no_grad():
        for i in range(5):
            o=m(inputs_embeds=e['embeddings'][i:i+1].unsqueeze(0),position_ids=torch.tensor([[i]]),past_key_values=cache,use_cache=True,output_hidden_states=True,return_dict=True)
            cache=o.past_key_values;refs.append(dict(position=i,logits=m.lm_head(o.hidden_states[-1])[:,-1].clone()))
    e['logit_proofs']=refs
    assert whole_sequence_proof(SimpleNamespace(llm=m),e)['passed']
    wrong=dict(e);wrong['embeddings']=e['embeddings'].roll(1,0)
    assert not whole_sequence_proof(SimpleNamespace(llm=m),wrong)['passed']
