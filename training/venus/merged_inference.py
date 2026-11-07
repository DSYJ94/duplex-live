"""Disposable inference-only LoRA folding; never a training checkpoint transform.

The unmerged forward adds separately rounded outputs. Folding instead rounds
an effective weight once, so BF16 bit equivalence is NOT claimed. Native speech,
behavior and latency must be requalified. Originals/Adam/RNG remain untouched.
"""
import math
import torch
from torch import nn
from .lora import LoRALinear


def merge_for_inference(model):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Only a frozen eval inference model may be folded')
    sites = [(name, module) for name,module in model.named_modules() if isinstance(module,LoRALinear)]
    if not sites:
        raise ValueError('Explicit learned adapter required')
    for name,module in sites:
        if not name or not module.enabled or type(module.base) is not nn.Linear:
            raise ValueError('Enabled explicit Linear LoRA projections required')
        if module.base.weight.dtype not in (torch.float32,torch.bfloat16,torch.float16):
            raise ValueError('Explicit supported native storage dtype required')
        tensors=[module.base.weight,module.a,module.b]
        if hasattr(module,'extra_a'):
            tensors.extend([module.extra_a,module.extra_b])
        if not all(torch.isfinite(p).all() for p in tensors):
            raise ValueError('Nonfinite adapter/base projection')
        if not math.isfinite(module.alpha/module.rank) or (hasattr(module,'extra_a') and not math.isfinite(module.extra_scale)):
            raise ValueError('Finite original branch scales required')
    replacements=[]
    try:
        with torch.no_grad():
            for name,module in sites:
                parent_name,_,attribute=name.rpartition('.')
                parent=model.get_submodule(parent_name) if parent_name else model
                weight=module.base.weight.float() + (module.b @ module.a)*(module.alpha/module.rank)
                if hasattr(module,'extra_a'):
                    weight=weight+(module.extra_b @ module.extra_a)*module.extra_scale
                weight=weight.to(module.base.weight.dtype)
                if not torch.isfinite(weight).all():
                    raise ValueError('Folded projection is nonfinite after native dtype cast')
                # Meta construction initializes no real random values or tensor storage.
                merged=nn.Linear(module.base.in_features,module.base.out_features,
                                 bias=module.base.bias is not None,device='meta',dtype=weight.dtype)
                merged.weight=nn.Parameter(weight,requires_grad=False)
                if module.base.bias is not None:
                    merged.bias=nn.Parameter(module.base.bias.detach().clone(),requires_grad=False)
                merged.eval();replacements.append((parent,attribute,module))
                setattr(parent,attribute,merged)
    except BaseException:
        for parent,attribute,original in replacements:
            setattr(parent,attribute,original)
        raise
    return dict(enabled=True,version='disposable-lora-fold-v1',modules=[name for name,_ in sites],
                projections=len(sites),weight_accumulation='float32',
                weight_storage_dtypes=sorted({str(module.base.weight.dtype) for _,module in sites}),
                original_checkpoint_changed=False,optimizer_state_changed=False,
                bit_equivalence_claimed=False,new_optimizer_updates=0,
                experimental=True,native_behavior_qualified=False)
