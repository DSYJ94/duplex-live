"""Small explicit LoRA module; frozen native weights, named portable adapter state."""
from contextlib import contextmanager
import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=16):
        super().__init__()
        self.base = base
        self.rank, self.alpha, self.enabled = rank, alpha, True
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x):
        original = self.base(x)
        if not self.enabled:
            return original
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.a), self.b)
        return original + (delta * (self.alpha / self.rank)).to(original.dtype)


def install(llm, rank=16, alpha=16):
    llm.requires_grad_(False)
    names = []
    for layer_id, layer in enumerate(llm.model.layers):
        for name in ('q_proj', 'v_proj'):
            base = getattr(layer.self_attn, name)
            if not isinstance(base, nn.Linear):
                raise TypeError('Only explicit original Linear projections may be adapted')
            setattr(layer.self_attn, name, LoRALinear(base, rank, alpha))
            names.append(f'model.layers.{layer_id}.self_attn.{name}')
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


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_adapter(model, state):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(state):
        raise ValueError('Adapter parameter names do not match the exact architecture')
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value.to(params[name]))


def selected_forward(llm, example, device):
    """Position i predicts the next native token; no user/framing token is labelled."""
    x = example['embeddings'].to(device).unsqueeze(0).detach().requires_grad_(True)
    positions = torch.arange(x.shape[1], device=device).unsqueeze(0)
    output = llm.model(inputs_embeds=x, position_ids=positions, use_cache=False, return_dict=True)
    hidden = output.last_hidden_state[0, example['positions'].to(device)]
    return llm.lm_head(hidden).float(), hidden.float()


def scene_loss(llm, example, device, kl_weight=.1, hidden_weight=.01):
    logits, hidden = selected_forward(llm, example, device)
    target = example['targets'].to(device)
    losses = torch.nn.functional.cross_entropy(logits, target, reduction='none')
    # Equal contribution per labelled decision/text group, then per scene.
    decision = example['decision_mask'].to(device)
    groups = [losses[m].mean() for m in (decision, ~decision) if m.any()]
    ce = torch.stack(groups).mean()
    kl, retention = ce.new_zeros(()), ce.new_zeros(())
    if example['family'] == 'ordinary':
        with torch.no_grad(), disabled(llm):
            base_logits, base_hidden = selected_forward(llm, example, device)
        kl = torch.nn.functional.kl_div(logits.log_softmax(-1), base_logits.softmax(-1), reduction='batchmean')
        retention = (hidden - base_hidden).square().mean()
    loss = ce + kl_weight * kl + hidden_weight * retention
    return loss, dict(ce=float(ce.detach()), kl=float(kl.detach()), hidden=float(retention.detach()))
