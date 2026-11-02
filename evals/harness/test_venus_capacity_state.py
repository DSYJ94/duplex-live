import copy
from pathlib import Path
import pytest
import torch
from torch import nn
from training.venus.lora import install,adapter_parameters
from training.venus.capacity_state import expand_model,expand_adam_state,expand_opal_state


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.model=nn.Module();self.model.layers=nn.ModuleList()
        for _ in range(3):
            layer=nn.Module();layer.self_attn=nn.Module();layer.self_attn.q_proj=nn.Linear(8,8);layer.self_attn.v_proj=nn.Linear(8,8);self.model.layers.append(layer)
    def forward(self,x):
        for layer in self.model.layers:x=torch.tanh(layer.self_attn.q_proj(x)+layer.self_attn.v_proj(x))
        return x


def trained():
    torch.manual_seed(42);m=Model();install(m,rank=2,alpha=2,last_layers=1)
    opt=torch.optim.AdamW(list(adapter_parameters(m).values()),lr=.01)
    x=torch.arange(16,dtype=torch.float32).reshape(2,8)/16
    m(x).square().mean().backward();opt.step();opt.zero_grad()
    return m,opt,x


def test_expansion_preserves_outputs_objects_and_rng_and_old_adam_next_step():
    m,opt,x=trained();reference=copy.deepcopy(m);old=adapter_parameters(m);saved=copy.deepcopy(opt.state_dict());before=m(x).detach();rng=torch.get_rng_state().clone()
    spec=expand_model(m,last_layers=3,rank=2,alpha=2,seed=123)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(before,m(x))
    params=adapter_parameters(m);assert all(params[n] is p for n,p in old.items())
    mapped=expand_adam_state(saved,spec['before_names'],params)
    new_opt=torch.optim.AdamW(list(params.values()),lr=.01);new_opt.load_state_dict(mapped)
    for ident,name in enumerate(params):assert int(mapped['state'][ident]['step'])==(1 if name in old else 0)
    ref_opt=torch.optim.AdamW(list(adapter_parameters(reference).values()),lr=.01);ref_opt.load_state_dict(saved)
    reference(x).square().mean().backward();ref_opt.step()
    m(x).square().mean().backward();new_opt.step()
    assert all(torch.equal(p,adapter_parameters(reference)[n]) for n,p in params.items() if n in old)
    assert all(int(new_opt.state[p]['step'])==1 for n,p in params.items() if n not in old)
    assert saved['param_groups'][0]['params']==[0,1,2,3]


def test_invalid_existing_configuration_fails_before_mutating_model():
    m,_,_=trained();old=adapter_parameters(m)
    with pytest.raises(ValueError):expand_model(m,last_layers=3,rank=4,alpha=4,seed=123)
    assert adapter_parameters(m).keys()==old.keys()


def test_real_upstream_opal_complete_expanded_state_restores(tmp_path):
    from training.venus.opal_adapter import wrap,Options
    from training.venus.native_optimizer import assert_optimizer_equal
    source=Path('/private/tmp/duplex-opal-upstream-20261028')
    if not (source/'.git').exists():pytest.skip('Pinned upstream checkout not available')
    m,base,x=trained()
    old_runtime=wrap(base,m,source=source,run_dir=tmp_path/'old',options=Options(total_steps=8,goal='Fixture restore proof'))
    try:
        m(x).square().mean().backward();old_runtime.step(loss=m(x).square().mean())
        state=copy.deepcopy(old_runtime.state_dict())
    finally:old_runtime.close()
    spec=expand_model(m,last_layers=3,rank=2,alpha=2,seed=123);params=adapter_parameters(m)
    expanded=expand_opal_state(state,spec['before_names'],params)
    for key in set(state)-{'updater','devices','user_optimizers'}:
        from training.venus.state import exact_state
        assert exact_state(state[key],expanded[key])
    base2=torch.optim.AdamW(list(params.values()),lr=.01)
    program=tmp_path/'initial-program';program.mkdir()
    for name,text in expanded['files'].items():(program/name).write_text(text)
    runtime=wrap(base2,m,source=source,run_dir=tmp_path/'restored',options=Options(total_steps=8,goal='Fixture restore proof'),initial_program=program)
    try:
        runtime.load_state_dict(expanded)
        assert_optimizer_equal(expanded,runtime.state_dict())
    finally:runtime.close()
