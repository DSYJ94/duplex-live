"""Semantic LoRA only after the released speech/control branch points."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


class SemanticLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base=base
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.normal_(self.a,std=.02)

    def forward(self, value):
        original=self.base(value)
        delta=F.linear(F.linear(value.float(),self.a),self.b)
        return original+delta.to(original.dtype)


def attach(model, rank=8):
    if (rank<=0 or model.config.text_config.num_hidden_layers!=28
            or model.stoken_model_index!=24 or model.control_model_index!=20):
        raise RuntimeError('Semantic adapter requires the pinned native branch points')
    model.requires_grad_(False);modules={}
    device=next(model.parameters()).device
    with torch.random.fork_rng(devices=[device.index]):
        for index in range(24,28):
            layer=model.model.layers[index]
            for group,names in (('self_attn',('q_proj','k_proj','v_proj','o_proj')),
                    ('mlp',('gate_proj','up_proj','down_proj'))):
                parent=getattr(layer,group)
                for name in names:
                    base=getattr(parent,name)
                    if not isinstance(base,nn.Linear):raise RuntimeError('Unexpected or already adapted native projection')
                    adapted=SemanticLinear(base,rank);setattr(parent,name,adapted)
                    modules[f'model.layers.{index}.{group}.{name}']=adapted
    model.eval()
    expected={id(p) for module in modules.values() for p in (module.a,module.b)}
    if {id(p) for p in model.parameters() if p.requires_grad}!=expected:
        raise RuntimeError('Unexpected trainable native parameter')
    return modules


def detach(model, modules):
    for name,module in modules.items():
        parts=name.split('.');parent=model
        for part in parts[:-1]:parent=getattr(parent,part)
        if getattr(parent,parts[-1]) is not module:raise RuntimeError('Adapter topology changed')
        setattr(parent,parts[-1],module.base)


def state_dict(modules):
    return {name+'.'+key:getattr(module,key).detach().cpu().clone()
        for name,module in modules.items() for key in ('a','b')}


def load_state(modules, state):
    expected={name+'.'+key for name in modules for key in ('a','b')}
    if set(state)!=expected:raise RuntimeError('Missing or extra semantic adapter tensors')
    with torch.no_grad():
        for name,module in modules.items():
            for key in ('a','b'):
                value=state[name+'.'+key];parameter=getattr(module,key)
                if value.shape!=parameter.shape or not torch.isfinite(value).all():raise RuntimeError('Invalid semantic adapter tensor')
                parameter.copy_(value)


def frozen_hash(model):
    digest=hashlib.sha256()
    for name,parameter in model.named_parameters():
        if parameter.requires_grad:continue
        digest.update(name.encode())
        array=parameter.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(array))
    return digest.hexdigest()
