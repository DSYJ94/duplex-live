import copy
from pathlib import Path
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from training.venus.lora import install, adapter_parameters, ALL_PROJECTIONS
from training.venus.large_capacity import expand_broad
from training.venus.capacity_state import expand_adam_state
from training.venus.capacity_audit import audit


def parent():
    torch.manual_seed(82)
    config=Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      vocab_size=48, attention_dropout=0.)
    config._attn_implementation='eager'
    model=Qwen3ForCausalLM(config).eval();install(model,rank=2,alpha=2,last_layers=1)
    opt=torch.optim.AdamW(list(adapter_parameters(model).values()),lr=.003)
    x=torch.tensor([[2,3,4,5,6]])
    model(input_ids=x,labels=x).loss.backward();opt.step();opt.zero_grad()
    return model,opt,x


def test_large_rank_and_mlp_expansion_preserves_trained_function_rng_and_old_adam():
    model,opt,x=parent();before=model(input_ids=x).logits.detach().clone()
    old=adapter_parameters(model);old_weights={n:p.detach().clone() for n,p in old.items()}
    state=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
    spec=expand_broad(model,last_layers=3,rank=8,alpha=8,seed=20261104)
    # Splitting inherited rank2 + new rank6 must equal uniform rank8,
    # rather than silently counting the inherited basis a second time.
    assert spec['parameters']==3*8*(64+48+48+64+96+96+96)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(before,model(input_ids=x).logits)
    params=adapter_parameters(model)
    assert all(params[n] is p and torch.equal(p,old_weights[n]) for n,p in old.items())
    assert any('.mlp.gate_proj.' in n for n in spec['added_names'])
    assert any('.extra_a' in n for n in spec['added_names'])
    mapped=expand_adam_state(state,spec['before_names'],params)
    lookup=dict(zip(spec['before_names'],state['param_groups'][0]['params']))
    for i,(name,p) in enumerate(params.items()):
        for key in ('step','exp_avg','exp_avg_sq'):
            if name in old:assert torch.equal(mapped['state'][i][key],state['state'][lookup[name]][key])
            else:assert torch.count_nonzero(mapped['state'][i][key])==0
    expanded=torch.optim.AdamW(list(params.values()),lr=.003);expanded.load_state_dict(mapped)
    model(input_ids=x,labels=x).loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
    # Zero-B initialization makes A's first gradient zero; every new B must
    # receive a real gradient, then every new A can learn on the second update.
    assert all(torch.count_nonzero(params[n].grad)>0 for n in spec['added_names'] if n.endswith(('.b','.extra_b')))
    expanded.step();expanded.zero_grad();model(input_ids=x,labels=x).loss.backward()
    assert all(torch.count_nonzero(params[n].grad)>0 for n in spec['added_names'] if n.endswith(('.a','.extra_a')))
    assert all(p.grad is None for n,p in model.named_parameters() if n not in params)


def test_invalid_rescale_or_shrink_fails_without_modifying_parent():
    model,_,_=parent();old=adapter_parameters(model)
    for rank,alpha in ((1,1),(8,4)):
        with pytest.raises(ValueError):expand_broad(model,last_layers=3,rank=rank,alpha=alpha,seed=2)
        assert adapter_parameters(model).keys()==old.keys()


def test_actual_config_counts_match_real_meta_qwen_modules():
    path=Path('runs/public-data-20261025/model-metadata/venus/Realtime-Venus-Audio/config.json')
    result=audit(path);assert result['intermediate_size']==12288
    config=Qwen3Config(hidden_size=4096,intermediate_size=12288,num_hidden_layers=1,
                      num_attention_heads=32,num_key_value_heads=8,head_dim=128,vocab_size=8)
    config._attn_implementation='eager'
    with torch.device('meta'):model=Qwen3ForCausalLM(config)
    assert sum(p.numel() for p in model.model.layers[0].parameters())==result['dense_parameters_per_block']==192946432
    spec=install(model,rank=256,alpha=256,last_layers=1,target_modules=ALL_PROJECTIONS)
    variants={x['name']:x for x in result['variants']}
    assert spec['parameters']*36==variants['all7_all36_rank256']['trainable_parameters']==698351616
    assert variants['dense_existing_last6']['new_foundation_parameters']==0
    assert variants['six_new_identity_blocks']['new_foundation_parameters']==1157678592


def test_broad_rank_branches_restore_complete_real_opal_state(tmp_path):
    from training.venus.opal_adapter import wrap, Options
    from training.venus.capacity_state import expand_opal_state
    from training.venus.native_optimizer import assert_optimizer_equal
    from training.venus.state import exact_state
    source=Path('/private/tmp/duplex-opal-upstream-20261028')
    if not (source/'.git').exists():pytest.skip('Pinned upstream checkout unavailable')
    model,base,x=parent()
    old_runtime=wrap(base,model,source=source,run_dir=tmp_path/'old',
                     options=Options(total_steps=8,goal='Broad capacity fixture'))
    try:
        loss=model(input_ids=x,labels=x).loss;loss.backward();old_runtime.step(loss=loss)
        state=copy.deepcopy(old_runtime.state_dict())
    finally:old_runtime.close()
    spec=expand_broad(model,last_layers=3,rank=8,alpha=8,seed=3)
    parameters=adapter_parameters(model)
    expanded=expand_opal_state(state,spec['before_names'],parameters)
    assert all(exact_state(state[k],expanded[k]) for k in set(state)-{'updater','devices','user_optimizers'})
    program=tmp_path/'program';program.mkdir()
    for name,text in expanded['files'].items():(program/name).write_text(text)
    opt=torch.optim.AdamW(list(parameters.values()),lr=.003)
    restored=wrap(opt,model,source=source,run_dir=tmp_path/'restored',
                  options=Options(total_steps=8,goal='Broad capacity fixture'),initial_program=program)
    try:
        restored.load_state_dict(expanded)
        assert_optimizer_equal(expanded,restored.state_dict())
    finally:restored.close()
