"""Explicit trainable coverage; frozen source weights are never changed here.

The caller must verify the released weights and preserve native objectives,
Adam/RNG/program state, independent restore and complete behavioral gates.
"""
from evals.personaplex_adapters import AdapterLinear,install_adapters,adapter_parameters


def configure(model,variant):
    if variant not in ('broad-lora64','selected-dense5'):raise ValueError('Unknown declared native capacity branch')
    if any(isinstance(m,AdapterLinear) for m in model.modules()):raise ValueError('Fresh released base required, not stacked adapters')
    if variant=='selected-dense5' and len(model.transformer.layers)<5:raise ValueError('Five existing temporal layers required')
    model.requires_grad_(False)
    if variant=='broad-lora64':
        modules=install_adapters(model,last_layers=len(model.transformer.layers),rank=64,scaling=2.)
        learned=adapter_parameters(model)
        if set(learned)!={n for n,p in model.named_parameters() if p.requires_grad}:raise ValueError('Unexpected adapter ownership')
        details=dict(rank=64,scaling=2.,modules=modules,raw_QKV_adapted=False,zero_B_initialization=True)
    else:
        for layer in model.transformer.layers[-5:]:layer.requires_grad_(True)
        model.text_linear.requires_grad_(True)
        learned={n:p for n,p in model.named_parameters() if p.requires_grad}
        details=dict(last_temporal_layers=5,text_head=True,new_model_weights=0,new_optimizer_required=True)
    return dict(variant=variant,trainable_parameters=sum(p.numel() for p in learned.values()),
        trainable_tensor_names=sorted(learned),details=details,optimizer_updates=0,
        native_gradient_and_state_admission=False,adopted=False)
