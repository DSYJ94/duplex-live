"""Small explicit LoRA module; frozen native weights, named portable adapter state."""
from contextlib import contextmanager
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=16, *, generator=None):
        super().__init__()
        self.base = base
        self.rank, self.alpha, self.enabled = rank, alpha, True
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5), generator=generator)
        self.base.requires_grad_(False)

    def forward(self, x):
        original = self.base(x)
        if not self.enabled:
            return original
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a), self.b)
        result = original + (delta * (self.alpha / self.rank)).to(original.dtype)
        if hasattr(self, 'extra_a'):
            extra = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.extra_a), self.extra_b)
            result = result + (extra * self.extra_scale).to(result.dtype)
        return result


ALL_PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def projection_sites(llm, last_layers, targets):
    """Resolve the full selection before any module or gradient flag changes."""
    layers = llm.model.layers
    if (last_layers is not None and (type(last_layers) is not int or not 1 <= last_layers <= len(layers))
            or not isinstance(targets, (tuple, list)) or not targets
            or len(set(targets)) != len(targets) or not set(targets) <= set(ALL_PROJECTIONS)):
        raise ValueError('Explicit valid layer suffix and unique attention/MLP projections required')
    sites = []
    for index, layer in enumerate(layers):
        if last_layers is not None and index < len(layers) - last_layers:
            continue
        for name in targets:
            parent_name = 'self_attn' if name in ALL_PROJECTIONS[:4] else 'mlp'
            parent = getattr(layer, parent_name)
            sites.append((f'model.layers.{index}.{parent_name}.{name}', parent, name, getattr(parent, name)))
    return sites


def install(llm, rank=16, alpha=16,last_layers=None, target_modules=('q_proj','v_proj')):
    sites = projection_sites(llm, last_layers, target_modules)
    if any(not isinstance(module, nn.Linear) for _, _, _, module in sites):
        raise TypeError('Only explicit original Linear projections may be adapted')
    llm.requires_grad_(False)
    names = []
    for path, parent, name, base in sites:
        setattr(parent, name, LoRALinear(base, rank, alpha))
        names.append(path)
    return dict(modules=names, rank=rank, alpha=alpha,
                parameters=sum(p.numel() for p in llm.parameters() if p.requires_grad))


@contextmanager
def disabled(model):
    modules = [m for m in model.modules() if isinstance(m, LoRALinear)]
    before = [m.enabled for m in modules]
    try:
        for m in modules:
            m.enabled = False
        yield
    finally:
        for m, enabled in zip(modules, before):
            m.enabled = enabled


def adapter_parameters(model):
    """Adapter identity is structural; inference freezes its gradients too."""
    return {prefix+'.'+name if prefix else name: getattr(module,name)
            for prefix,module in model.named_modules() if isinstance(module,LoRALinear)
            for name in ('a','b','extra_a','extra_b') if hasattr(module,name)}


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in adapter_parameters(model).items()}


def load_adapter(model, state):
    params = adapter_parameters(model)
    if set(params) != set(state):
        raise ValueError('Adapter parameter names do not match the exact architecture')
    if any(not torch.is_tensor(state[n]) or state[n].shape != p.shape
           or not torch.isfinite(state[n]).all() for n, p in params.items()):
        raise ValueError('Adapter shapes and finite values must match exactly')
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(params[name]))


def selected_forward(llm, example, device):
    """Position i predicts the next native token; no user/framing token is labelled."""
    if example.get('forward_path')=='native-segmented-bptt-v1':
        return native_segmented_forward(llm,example,device)
    if example.get('forward_path')=='native-frozen-prefix-v1':
        return suffix_forward(llm,example,device)
    x = example['embeddings'].to(device).unsqueeze(0).detach().requires_grad_(True)
    positions = torch.arange(x.shape[1], device=device).unsqueeze(0)
    output = llm.model(inputs_embeds=x, position_ids=positions, use_cache=False, return_dict=True)
    hidden = output.last_hidden_state[0, example['positions'].to(device)]
    return llm.lm_head(hidden).float(), hidden.float()


def native_segmented_forward(llm,example,device):
    """Actual native feed shapes with full autograd through the KV history.

    No prefix/cache detach, no truncated temporal credit assignment. eval mode
    preserves the native attention path; eval does NOT disable autograd.
    """
    if llm.training or getattr(llm,'is_gradient_checkpointing',False):
        raise ValueError('Native attention mode and unmodified KV autograd required')
    wanted={int(p):i for i,p in enumerate(example['positions'])};logits={};hidden={};cache=None;cursor=0
    for segment in example['segments']:
        start,n=segment['start'],segment['length'];assert start==cursor
        x=example['embeddings'][start:start+n].to(device).unsqueeze(0).detach().requires_grad_(True)
        positions=torch.arange(start,start+n,device=device).unsqueeze(0)
        output=llm.model(inputs_embeds=x,position_ids=positions,past_key_values=cache,use_cache=True,return_dict=True)
        cache=output.past_key_values;assert cache is not None
        cursor+=n;last=cursor-1
        if last in wanted:
            # Preserve even the head's native GEMM shape, then take its last row.
            h=output.last_hidden_state;logits[last]=llm.lm_head(h)[0,-1].float();hidden[last]=h[0,-1].float()
    assert cursor==len(example['embeddings']) and set(logits)==set(wanted)
    return torch.stack([logits[int(p)] for p in example['positions']]),torch.stack([hidden[int(p)] for p in example['positions']])


def suffix_forward(llm,example,device):
    """Exact native suffix, with frozen lower-layer inputs and FULL suffix KV gradients."""
    from transformers.cache_utils import DynamicCache
    assert not llm.training and not getattr(llm,'is_gradient_checkpointing',False)
    start_layer=example['prefix_layers']
    if type(start_layer) is not int or not 0<=start_layer<len(llm.model.layers):
        raise ValueError('Explicit valid frozen prefix boundary required')
    # Frozen-prefix inputs were captured before these layers were adapted.
    # Reusing them for a wider adapter would silently bypass its computation
    # and gradients, even if the reported trainable parameter count increased.
    for layer in llm.model.layers[:start_layer]:
        if (torch.is_grad_enabled() and any(p.requires_grad for p in layer.parameters())) or any(
                isinstance(m,LoRALinear) and m.enabled for m in layer.modules()):
            raise ValueError('Frozen-prefix cache excludes trainable or adapted layers; rebuild compatible native histories')
    wanted={int(p) for p in example['positions']};logits={};hidden={};cache=DynamicCache();cursor=0
    for entry,segment in zip(example['prefix_inputs'],example['segments']):
        start,n=segment['start'],segment['length'];assert start==cursor
        # Lower layers are frozen: these inputs are constants, not truncated
        # states of any trainable computation. Suffix KV gradients stay intact.
        h=entry['hidden'].to(device).detach()
        kwargs={k:(v.to(device) if torch.is_tensor(v) else tuple(t.to(device) for t in v) if isinstance(v,tuple) else v) for k,v in entry['kwargs'].items()}
        kwargs.update(past_key_value=cache,use_cache=True,output_attentions=False)
        for layer in llm.model.layers[start_layer:]:
            output=layer(h,**kwargs)
            # Qwen3 decoder layers return a tuple in the pinned older runtime,
            # and the whole tensor in newer Transformers. Never drop batch0's
            # dimension by indexing a tensor as though it were a tuple.
            h=output[0] if isinstance(output,tuple) else output
        h=llm.model.norm(h);cursor+=n;last=cursor-1
        if last in wanted:logits[last]=llm.lm_head(h)[0,-1].float();hidden[last]=h[0,-1].float()
    assert len(example['prefix_inputs'])==len(example['segments']) and cursor==len(example['embeddings']) and set(logits)==wanted
    return torch.stack([logits[int(p)] for p in example['positions']]),torch.stack([hidden[int(p)] for p in example['positions']])


def scene_loss(llm, example, device, kl_weight=.1, hidden_weight=.01,
               action_token_ids=None, action_aux_weight=0.):
    # Opt-in preparation only: existing admitted training calls are unchanged.
    # A future phase must explicitly admit the revised objective and exact index.
    if type(action_aux_weight) not in (int,float) or not math.isfinite(action_aux_weight) or not 0 <= action_aux_weight <= .5:
        raise ValueError('Finite explicitly bounded action auxiliary weight required')
    if action_aux_weight and action_token_ids is None:
        raise ValueError('Action objective requires audited native target identities')
    selected = None
    if action_token_ids is not None:
        if example.get('fold') not in ('TRAIN','DEV'):
            raise ValueError('Action training diagnostics exclude held-out targets')
        from .action_targets import masks
        selected, _ = masks(example['targets'], example['decision_mask'], action_token_ids)
    logits, hidden = selected_forward(llm, example, device)
    target = example['targets'].to(device)
    losses = torch.nn.functional.cross_entropy(logits, target, reduction='none')
    # Equal contribution per labelled decision/text group, then per scene.
    decision = example['decision_mask'].to(device)
    decision_ce=losses[decision].mean() if decision.any() else None
    text_ce=losses[~decision].mean() if (~decision).any() else None
    groups=[v for v in (decision_ce,text_ce) if v is not None]
    ce = torch.stack(groups).mean()
    kl, retention = ce.new_zeros(()), ce.new_zeros(())
    if example['family'] == 'ordinary':
        with torch.no_grad(), disabled(llm):
            base_logits, base_hidden = selected_forward(llm, example, device)
        kl = torch.nn.functional.kl_div(logits.log_softmax(-1), base_logits.softmax(-1), reduction='batchmean')
        retention = (hidden - base_hidden).square().mean()
    loss = ce + kl_weight * kl + hidden_weight * retention
    parts = dict(ce=float(ce.detach()), kl=float(kl.detach()), hidden=float(retention.detach()),decision_ce=float(decision_ce.detach()) if decision_ce is not None else None,text_ce=float(text_ce.detach()) if text_ce is not None else None)
    if selected is not None:
        from .action_targets import metrics
        parts.update(metrics(logits, target, selected, losses, control_token_ids=action_token_ids))
        action_mask = selected['action'].to(device)
        auxiliary = losses[action_mask].mean() if action_mask.any() else loss.new_zeros(())
        if action_aux_weight:
            loss = loss + action_aux_weight * auxiliary
        parts.update(action_aux_weight=action_aux_weight, action_auxiliary=float(auxiliary.detach()))
    return loss, parts
