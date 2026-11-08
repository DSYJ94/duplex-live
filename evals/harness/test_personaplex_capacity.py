import copy
import torch
import pytest
from torch import nn
from training.personaplex.capacity import configure
from evals.personaplex_adapters import install_adapters


class Gating(nn.Module):
    def __init__(self):
        super().__init__();self.linear_in=nn.Linear(4,16,bias=False);self.linear_out=nn.Linear(8,4,bias=False);self.activation=nn.SiLU()
    def forward(self,x):
        a,b=self.linear_in(x).chunk(2,-1);return self.linear_out(self.activation(a)*b)


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.transformer=nn.Module();self.transformer.layers=nn.ModuleList([Gating() for _ in range(6)]);self.text_linear=nn.Linear(4,12,bias=False)
    def forward(self,x):
        for layer in self.transformer.layers:x=x+layer(x)
        return self.text_linear(x)


def test_broad_zero_init_preserves_logits_and_all_frozen_weights_then_backprops():
    torch.manual_seed(12);model=Model();x=torch.randn(1,3,4);original=copy.deepcopy(model.state_dict());before=model(x).detach()
    report=configure(model,'broad-lora64')
    torch.testing.assert_close(model(x),before,rtol=0,atol=0)
    assert report['details']['rank']==64 and len(report['details']['modules'])==13
    for name,tensor in original.items():torch.testing.assert_close(model.state_dict()[name],tensor,rtol=0,atol=0)
    model(x).square().sum().backward()
    b=[p for n,p in model.named_parameters() if 'lora_B' in n]
    assert b and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in b)
    with pytest.raises(ValueError,match='stacked'):configure(model,'broad-lora64')


def test_selected_dense_changes_only_ownership_not_values():
    model=Model();old=copy.deepcopy(model.state_dict());report=configure(model,'selected-dense5')
    assert not any(p.requires_grad for p in model.transformer.layers[0].parameters())
    assert all(p.requires_grad for l in model.transformer.layers[1:] for p in l.parameters())
    assert model.text_linear.weight.requires_grad and report['details']['new_model_weights']==0
    for name,value in old.items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)


def test_default_legacy_rank_is_still_eight():
    model=Model();install_adapters(model,last_layers=4)
    assert all(m.lora_A.out_features==8 for m in model.modules() if hasattr(m,'lora_A'))
