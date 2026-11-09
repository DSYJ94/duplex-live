"""Full named PersonaPlex adapters, Adam/OPAL/programs and RNG; no updates.

PersonaPlex's AdapterLinear is different from the Venus LoRALinear. Never use
the latter's state selector here: it would silently select zero coefficients.
"""
import os
from pathlib import Path


def adapter_state(model):
    from evals.personaplex_adapters import adapter_parameters
    selected=adapter_parameters(model)
    if not selected:raise ValueError('Actual named PersonaPlex adapters required')
    return {n:p.detach().cpu().clone() for n,p in selected.items()}


def load_adapter(model,state):
    import torch
    from evals.personaplex_adapters import adapter_parameters
    selected=adapter_parameters(model)
    if not selected or selected.keys()!=state.keys():raise ValueError('Full exact PersonaPlex adapter membership required')
    for name,p in selected.items():
        value=state[name]
        if (not torch.is_tensor(value) or value.shape!=p.shape or value.dtype!=p.dtype
                or not torch.isfinite(value).all()):raise ValueError('Invalid native adapter: '+name)
    with torch.no_grad():
        for name,p in selected.items():p.copy_(state[name].to(device=p.device))


def materialize_adam(base):
    """Allocate real empty Adam moments without any dummy optimizer update."""
    import torch
    if base.state:raise ValueError('Only a fresh zero-update Adam can be materialized')
    for group in base.param_groups:
        if group['amsgrad'] or group['capturable'] or group.get('fused'):
            raise ValueError('Explicit nonfused noncapturable AdamW state format required')
        for p in group['params']:
            if p.dtype!=torch.float32 or not p.requires_grad:
                raise ValueError('Materialize only actual FP32 trainable adapters')
            base.state[p]=dict(step=torch.tensor(0.,dtype=torch.float32),
                exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
    return dict(parameter_tensors=len(base.state),optimizer_updates=0,
        moment_elements=sum(2*p.numel() for p in base.state))


def save(path,model,runtime,metadata,references):
    import torch
    from training.venus.state import rng_state
    from evals.moshi_model_download import sha256
    from training.venus.opal_adapter import checkpoint_manifest
    path=Path(path);program=checkpoint_manifest(runtime.optimizer)
    state=dict(schema='dl-moshi-matched-state-v1',metadata=metadata,
        adapter=adapter_state(model),optimizer=runtime.optimizer.state_dict(),
        rng=rng_state(),step=program['update'],diagnostics=runtime.diagnostics_state,
        references=references)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(state,tmp);os.replace(tmp,path)
    return sha256(path)


def restore(state,model,runtime,expected):
    from training.venus.native_optimizer import assert_optimizer_equal
    from training.venus.state import exact_state,restore_rng,rng_state
    if state['schema']!='dl-moshi-matched-state-v1' or state['metadata']!=expected:
        raise ValueError('Exact source/data/base/capacity/recipe identity required')
    if state['step']!=state['optimizer']['update']:
        raise ValueError('Never replay a saved optimizer update')
    load_adapter(model,state['adapter'])
    runtime.optimizer.load_state_dict(state['optimizer'])
    runtime.restore_diagnostics(state['diagnostics']);restore_rng(state['rng'])
    assert_optimizer_equal(state['optimizer'],runtime.optimizer.state_dict())
    if not exact_state(state['adapter'],adapter_state(model)) or not exact_state(state['rng'],rng_state()):
        raise ValueError('Fresh adapter or RNG restore differs')
    return state['step']
