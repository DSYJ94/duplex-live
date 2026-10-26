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


def test_native_candidate_loading_preserves_trained_outputs_and_freezes_weights(tmp_path):
    from training.venus.inference import load_candidate, MODEL_REVISION
    m=model();e=example(m);install(m,2,2,last_layers=1)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
    loss,_=scene_loss(m,e,'cpu');loss.backward();opt.step();opt.zero_grad()
    expected=selected_forward(m,e,'cpu')[0].detach()
    path=tmp_path/'step-0001.pt'
    metadata=dict(model_revision=MODEL_REVISION,source='a'*40,recipe=dict(rank=2,alpha=2,
        last_layers=1,projections=['q_proj','v_proj'],forward_path='native-frozen-prefix-v1',updates=38))
    save(path,m,opt,1,metadata)
    fresh=model();profile=load_candidate(fresh,path)
    assert torch.equal(expected,selected_forward(fresh,e,'cpu')[0])
    assert not any(p.requires_grad for p in fresh.parameters())
    assert profile['optimizer_updates']==1 and profile['planned_updates']==38
    assert profile['experimental_candidate'] and not profile['adopted'] and not profile['native_behavior_qualified']
    proof=tmp_path/'expected-step-0002-proof-only.pt';proof.write_bytes(path.read_bytes())
    with pytest.raises(ValueError,match='proof-only'):load_candidate(model(),proof)
    metadata['model_revision']='wrong';save(path,m,opt,1,metadata)
    with pytest.raises(ValueError,match='revision'):load_candidate(model(),path)


def test_adapter_loader_rejects_broadcastable_shapes_and_nonfinite_values():
    from training.venus.lora import load_adapter
    m=model();install(m,2,2,last_layers=1);state=adapter_state(m);name=next(iter(state))
    bad=dict(state);bad[name]=state[name][:1]
    with pytest.raises(ValueError,match='shapes'):load_adapter(m,bad)
    bad=dict(state);bad[name]=state[name].clone();bad[name][0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):load_adapter(m,bad)


def test_native_segment_to_training_alignment_is_red_capable():
    from types import SimpleNamespace
    from training.venus.native_cache import training_input_proof
    m=model();e=example(m);e['positions']=torch.tensor([0,1,2,3,4]);e['targets']=torch.tensor([2,3,4,5,6])
    refs=[];cache=None
    with torch.no_grad():
        for i in range(5):
            o=m(inputs_embeds=e['embeddings'][i:i+1].unsqueeze(0),position_ids=torch.tensor([[i]]),past_key_values=cache,use_cache=True,output_hidden_states=True,return_dict=True)
            cache=o.past_key_values;refs.append(dict(position=i,logits=m.lm_head(o.hidden_states[-1])[:,-1].clone()))
    e['logit_proofs']=refs
    assert training_input_proof(SimpleNamespace(llm=m),e)['passed']
    wrong=dict(e);wrong['embeddings']=e['embeddings'].roll(1,0)
    assert not training_input_proof(SimpleNamespace(llm=m),wrong)['passed']


def test_native_segmented_gradient_preserves_full_history_credit():
    from training.venus.lora import native_segmented_forward
    m=model();e=example(m);install(m,2,2)
    with torch.no_grad():
        for name,p in m.named_parameters():
            if p.requires_grad and name.endswith('.b'):p.normal_(std=.03)
    e['segments']=[dict(start=i,length=1,return_logits=True) for i in range(6)]
    logits,_=selected_forward(m,e,'cpu');loss=torch.nn.functional.cross_entropy(logits,e['targets']);loss.backward()
    expected={n:p.grad.clone() for n,p in m.named_parameters() if p.requires_grad}
    m.zero_grad();native,_=native_segmented_forward(m,e,'cpu')
    assert torch.allclose(logits,native,atol=1e-6,rtol=1e-6)
    torch.nn.functional.cross_entropy(native,e['targets']).backward()
    assert all(torch.allclose(p.grad,expected[n],atol=1e-6,rtol=1e-5) for n,p in m.named_parameters() if p.requires_grad)


def test_admission_uses_full_epochs_and_never_drops_train_scenes():
    from collections import Counter
    from training.venus.train import RECIPE,order,admit_epochs
    before=dict(RECIPE)
    try:
        admission=admit_epochs(10,20,2000)
        assert admission['epochs']>=2 and admission['conservative_required_seconds']<2000
        RECIPE.update(epochs=admission['epochs'],updates=admission['updates'])
        rows=[dict(id=str(i),fold='TRAIN') for i in range(152)]+[dict(id='held',fold='HOLDOUT')]
        batches=order(rows);counts=Counter(r['id'] for b in batches for r in b)
        assert len(counts)==152 and set(counts.values())=={admission['epochs']}
        with pytest.raises(TimeoutError):admit_epochs(100,100,1000)
    finally:RECIPE.clear();RECIPE.update(before)


def test_saved_activation_offload_preserves_native_gradients():
    from training.venus.offload import ActivationOffload
    m=model();e=example(m);install(m,2,2)
    with torch.no_grad():
        for n,p in m.named_parameters():
            if p.requires_grad and n.endswith('.b'):p.normal_(std=.02)
    e['segments']=[dict(start=i,length=1,return_logits=True) for i in range(6)]
    e['forward_path']='native-segmented-bptt-v1'
    loss,_=scene_loss(m,e,'cpu');loss.backward();expected={n:p.grad.clone() for n,p in m.named_parameters() if p.requires_grad}
    m.zero_grad();hook=ActivationOffload(m,minimum_bytes=1,force_cpu_test=True)
    with hook.context():
        same,_=scene_loss(m,e,'cpu');assert torch.equal(loss,same);same.backward()
    assert hook.saved_bytes>0 and hook.tensors>0
    assert all(torch.equal(p.grad,expected[n]) for n,p in m.named_parameters() if p.requires_grad)


def test_frozen_prefix_suffix_is_exact_and_keeps_all_trainable_gradients():
    from training.venus.prefix_cache import capture_prefix
    m=model();e=example(m);e['segments']=[dict(start=i,length=1,return_logits=True) for i in range(6)]
    e['forward_path']='native-segmented-bptt-v1'
    prefix,proof=capture_prefix(m,e);assert proof['passed'] and proof['logit_error']==0 and proof['hidden_error']==0
    spec=install(m,2,2,last_layers=1);assert spec['parameters']==224
    with torch.no_grad():
        for n,p in m.named_parameters():
            if p.requires_grad and n.endswith('.b'):p.normal_(std=.03)
    full,_=scene_loss(m,e,'cpu');full.backward();expected={n:p.grad.clone() for n,p in m.named_parameters() if p.requires_grad}
    m.zero_grad();same,_=scene_loss(m,prefix,'cpu');assert torch.equal(full,same);same.backward()
    assert all(torch.allclose(p.grad,expected[n],atol=1e-7,rtol=1e-5) for n,p in m.named_parameters() if p.requires_grad)
