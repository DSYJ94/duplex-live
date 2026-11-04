"""Opt-in large attention+MLP adapter preparation; no training admission.

Retain the parent's original GEMM shapes, tensors and optimizer identities.
Additional rank uses an independent zero-output branch, not rank-padding the
trained matrices (which can change floating-point reduction order).
"""
import math
import torch
from torch import nn
from .lora import ALL_PROJECTIONS, LoRALinear, adapter_parameters, projection_sites


def expand_broad(llm, *, last_layers, rank, alpha, seed):
    if type(rank) is not int or rank < 1 or type(seed) is not int or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Explicit positive rank/scale and local initialization seed required')
    old = adapter_parameters(llm)
    if not old:
        raise ValueError('Preserved trained parent adapters required')
    sites = projection_sites(llm, last_layers, ALL_PROJECTIONS)
    retained = set()
    for path, _, _, module in sites:
        if isinstance(module, LoRALinear):
            if (not module.enabled or hasattr(module, 'extra_a') or rank < module.rank
                    or alpha / rank != module.alpha / module.rank):
                raise ValueError('Cannot shrink, rescale or repeatedly rewrite inherited adapter basis')
            retained.update((path + '.a', path + '.b'))
        elif not isinstance(module, nn.Linear):
            raise ValueError('Original attention/MLP Linear projections required')
        base = module.base if isinstance(module, LoRALinear) else module
        if base.weight.device.type not in ('cpu', 'cuda'):
            raise ValueError('Real CPU/CUDA preparation required; metadata counts are separate')
    if retained != set(old):
        raise ValueError('Expansion must include every inherited adapter')
    generators = {}
    for _, parent, name, module in sites:
        base = module.base if isinstance(module, LoRALinear) else module
        key = str(base.weight.device)
        if key not in generators:
            generators[key] = torch.Generator(device=base.weight.device).manual_seed(seed)
        if isinstance(module, LoRALinear):
            extra = rank - module.rank
            if extra:
                a = torch.empty(extra, base.in_features, device=base.weight.device, dtype=torch.float32)
                nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generators[key])
                module.extra_a = nn.Parameter(a)
                module.extra_b = nn.Parameter(torch.zeros(base.out_features, extra, device=base.weight.device, dtype=torch.float32))
                module.extra_scale = alpha / rank
        else:
            setattr(parent, name, LoRALinear(base, rank, alpha, generator=generators[key]))
    current = adapter_parameters(llm)
    assert all(current[name] is parameter for name, parameter in old.items())
    return dict(format='native-broad-adapter-v1', last_layers=last_layers,
                effective_rank=rank, effective_alpha=alpha, projections=list(ALL_PROJECTIONS),
                before_names=list(old), after_names=list(current),
                added_names=[name for name in current if name not in old],
                parameters=sum(p.numel() for p in current.values()),
                trained=False, native_qualified=False)
