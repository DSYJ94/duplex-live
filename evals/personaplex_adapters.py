"""Small native PersonaPlex temporal adapters, without changing frozen key names.

Adapt attention outputs and feed-forward linears in the final eight language
layers, plus the text head. NVIDIA uses a raw QKV parameter; do not silently
pretend a Linear replacement adapts it. Acoustic model parameters stay frozen.
"""
import math
from types import MethodType
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from evals.personaplex_reference import ReferenceBridge


class AdapterLinear(nn.Linear):
    def __init__(self, original, rank=8, scaling=2.):
        nn.Module.__init__(self)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight, self.bias = original.weight, original.bias
        self.scaling = scaling
        self.lora_A = nn.Linear(self.in_features, rank, bias=False, device=self.weight.device, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False, device=self.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, value):
        base = F.linear(value, self.weight, self.bias)
        update = self.lora_B(self.lora_A(value.to(self.lora_A.weight.dtype)))
        return base + update.to(base.dtype) * self.scaling


def install_adapters(model, last_layers=8):
    if not 1 <= last_layers <= len(model.transformer.layers):
        raise ValueError("Invalid native layer count")
    replaced = []

    def visit(module, prefix):
        for name, child in list(module.named_children()):
            path = prefix + "." + name
            if isinstance(child, AdapterLinear):
                raise RuntimeError("Adapters already installed")
            if isinstance(child, nn.Linear):
                setattr(module, name, AdapterLinear(child))
                replaced.append(path)
            else:
                visit(child, path)
        # Native ActivationGating calls F.linear on the two raw weights, so
        # replacing its children alone silently bypasses their adapters.
        if isinstance(getattr(module, "linear_in", None), AdapterLinear) and isinstance(
                getattr(module, "linear_out", None), AdapterLinear):
            def gating_forward(self, x):
                x = self.linear_in(x)
                batch, frames, _ = x.shape
                x = x.view(batch, frames, 2, -1)
                return self.linear_out(self.activation(x[..., 0, :]) * x[..., 1, :])
            module.forward = MethodType(gating_forward, module)

    first = len(model.transformer.layers)-last_layers
    for index in range(first, len(model.transformer.layers)):
        visit(model.transformer.layers[index], f"transformer.layers.{index}")
    if isinstance(model.text_linear, AdapterLinear):
        raise RuntimeError("Text head already adapted")
    model.text_linear = AdapterLinear(model.text_linear)
    replaced.append("text_linear")
    return replaced


def adapter_parameters(model):
    return {n:p for n,p in model.named_parameters() if ".lora_A." in n or ".lora_B." in n}


class GatedReferenceBridge(ReferenceBridge):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagonal = nn.Parameter(torch.zeros(self.active.shape[-1], device=self.active.device, dtype=self.active.dtype))

    def transform(self, reference):
        return reference * self.diagonal + super().transform(reference)


@contextmanager
def checkpoint_language_layers(model):
    """Checkpoint whole native layers only during unstreamed gradient training."""
    from torch.utils.checkpoint import checkpoint
    originals = []
    for layer in model.transformer.layers:
        original = layer.forward
        def forward(*args, _original=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, preserve_rng_state=False, **kwargs)
            return _original(*args, **kwargs)
        originals.append((layer, original))
        layer.forward = forward
    try:
        yield
    finally:
        for layer, original in originals:
            layer.forward = original
