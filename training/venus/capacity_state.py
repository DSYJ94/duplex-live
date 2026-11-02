"""Explicit wider-adapter phase preparation; no training/adoption authority.

Parent checkpoints remain immutable. Existing tensors/moments/counters and OPAL
history are retained; only new parameter identities receive zero Adam state.
"""
import copy
import torch
from torch import nn
from .lora import LoRALinear,adapter_parameters


def expand_model(llm,*,last_layers,rank,alpha,seed):
    layers=llm.model.layers
    if (type(last_layers) is not int or not 1<=last_layers<=len(layers)
            or type(seed) is not int or type(rank) is not int or rank<1):
        raise ValueError('Explicit valid wider adapter configuration required')
    before=adapter_parameters(llm)
    if not before:raise ValueError('Existing parent adapter required')
    selected=[];retained=set()
    for index,layer in enumerate(layers):
        for projection in ('q_proj','v_proj'):
            module=getattr(layer.self_attn,projection)
            name=f'model.layers.{index}.self_attn.{projection}'
            if isinstance(module,LoRALinear):
                if index<len(layers)-last_layers or module.rank!=rank or module.alpha!=alpha or not module.enabled:
                    raise ValueError('Cannot shrink, rescale or change the existing parent adapter')
                retained.update((name+'.a',name+'.b'))
            elif index>=len(layers)-last_layers:
                if not isinstance(module,nn.Linear) or module.weight.device.type not in ('cpu','cuda'):
                    raise ValueError('Original CPU/CUDA linear projections required')
                selected.append((index,layer,projection,module))
    if retained!=set(before) or not selected:raise ValueError('Strict parameter-set expansion required')
    # Local generators initialize new A matrices without consuming the saved
    # training RNG. New B matrices are zero; existing module objects stay exact.
    generators={}
    for _,layer,projection,base in selected:
        key=str(base.weight.device)
        if key not in generators:
            generators[key]=torch.Generator(device=base.weight.device).manual_seed(seed)
        setattr(layer.self_attn,projection,LoRALinear(base,rank,alpha,generator=generators[key]))
    after=adapter_parameters(llm)
    assert all(after[n] is p for n,p in before.items())
    return dict(before_names=list(before),after_names=list(after),
                added_names=[n for n in after if n not in before],
                parameters=sum(p.numel() for p in after.values()))


def expand_adam_state(state,old_names,parameters):
    """One existing AdamW group, remapped by explicit parameter identity."""
    names=list(parameters);groups=state['param_groups']
    if (len(groups)!=1 or len(set(old_names))!=len(old_names) or not set(old_names)<set(names)
            or len(groups[0]['params'])!=len(old_names) or groups[0].get('capturable') or groups[0].get('fused')):
        raise ValueError('Explicit strict expansion of one ordinary AdamW group required')
    old_ids=groups[0]['params']
    if len(set(old_ids))!=len(old_ids) or set(state['state'])!=set(old_ids):
        raise ValueError('Complete existing Adam parameter state required')
    result=copy.deepcopy(state);result['state']={};result['param_groups'][0]['params']=list(range(len(names)))
    old=dict(zip(old_names,old_ids))
    for ident,name in enumerate(names):
        p=parameters[name]
        if name in old:
            value=state['state'][old[name]]
            if set(value)!={'step','exp_avg','exp_avg_sq'} or any(value[k].shape!=p.shape for k in ('exp_avg','exp_avg_sq')):
                raise ValueError('Exact parent AdamW tensor shapes required')
            result['state'][ident]=copy.deepcopy(value)
        else:
            result['state'][ident]=dict(step=torch.tensor(0.,dtype=torch.float32),exp_avg=torch.zeros_like(p,device='cpu'),exp_avg_sq=torch.zeros_like(p,device='cpu'))
    return result


def expand_opal_state(state,old_names,parameters):
    """Versioned derivative; all non-parameter OPAL/scheduler history is exact."""
    if (state.get('opal')!=1 or len(state['user_optimizers'])!=1
            or len(state['updater']['opts'])!=1 or len(state['devices']['opts'])!=1):
        raise ValueError('Exact single-optimizer wrapped OPAL state required')
    result=copy.deepcopy(state)
    for source,target in ((state['user_optimizers'],result['user_optimizers']),
                          (state['updater']['opts'],result['updater']['opts'])):
        target[0]=expand_adam_state(source[0],old_names,parameters)
    old_ids=state['user_optimizers'][0]['param_groups'][0]['params']
    devices=state['devices']['opts'][0]
    if (len(devices['param_groups'])!=1 or set(devices['state'])!=set(old_ids)
            or devices['param_groups'][0]['params']!=[None]*len(old_ids)):
        raise ValueError('Complete parent device map required')
    mapped=copy.deepcopy(devices);mapped['state']={}
    lookup=dict(zip(old_names,old_ids))
    for ident,(name,p) in enumerate(parameters.items()):
        mapped['state'][ident]=(copy.deepcopy(devices['state'][lookup[name]]) if name in lookup else
            dict(step='cpu',exp_avg=p.device.type,exp_avg_sq=p.device.type))
    mapped['param_groups'][0]['params']=[None]*len(parameters)
    result['devices']['opts'][0]=mapped
    return result
