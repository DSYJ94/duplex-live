"""Explicit FP32 context adapters; preserve every original parameter identity/name."""
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'dl-controller-context-adapter-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContextLinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        if type(base) is not nn.Linear or any(p.requires_grad for p in base.parameters()):
            raise ValueError('Context projection requires an original frozen Linear')
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.lora_A = nn.Parameter(torch.empty(8, self.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, 8, dtype=torch.float32, device=base.weight.device))
        nn.init.normal_(self.lora_A, std=.02)

    def forward(self, hidden):
        original = F.linear(hidden, self.weight, self.bias)
        update = F.linear(F.linear(hidden.float(), self.lora_A), self.lora_B) * 2.
        return original + update.to(original.dtype)


def install(model, seed=42424242):
    source = dict(model.named_parameters())
    if any(p.requires_grad for p in source.values()):
        raise ValueError('Source must be fully frozen before installing context adapters')
    layers = model.model.language_model.layers
    if len(layers) != 32:
        raise ValueError('Expected the pinned 32-layer decoder')
    names = []
    # Construction is seed-controlled without advancing the caller's CPU/CUDA RNG.
    devices = sorted({p.device.index for p in source.values() if p.device.type == 'cuda'})
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for index in range(28, 32):
            layer = layers[index]
            attention = 'linear_attn.out_proj' if layer.layer_type == 'linear_attention' else 'self_attn.o_proj'
            for suffix in (attention, 'mlp.down_proj'):
                name = f'model.language_model.layers.{index}.{suffix}'
                parent_name, attr = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, attr, ContextLinear(getattr(parent, attr)))
                names.append(name)
    after = dict(model.named_parameters())
    if any(after.get(n) is not p or p.requires_grad for n, p in source.items()):
        raise ValueError('An original parameter was replaced, renamed or unfrozen')
    adapters = {n: p for n, p in after.items() if n not in source}
    expected = {name + '.' + side for name in names for side in ('lora_A', 'lora_B')}
    if set(adapters) != expected or {id(p) for p in model.parameters() if p.requires_grad} != {id(p) for p in adapters.values()}:
        raise ValueError('Unexpected optimizer partition')
    return adapters


def save(folder, adapters, assets_sha256, provenance):
    from safetensors.torch import save_file
    folder = Path(folder)
    folder.mkdir(exist_ok=False)
    tensors = {n: p.detach().cpu().contiguous() for n, p in adapters.items()}
    save_file(tensors, str(folder / 'adapter.safetensors'))
    metadata = dict(schema=SCHEMA, base_assets_sha256=assets_sha256, rank=8, scaling=2.,
                    implementation_sha256=sha(__file__), weights_sha256=sha(folder / 'adapter.safetensors'),
                    tensor_shapes={n: list(p.shape) for n, p in tensors.items()}, provenance=provenance,
                    production_adopted=False, native_voice_weights_changed=False)
    (folder / 'adapter.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def restore(folder, adapters, assets_sha256):
    from safetensors.torch import load_file
    folder = Path(folder)
    meta = json.loads((folder / 'adapter.json').read_text())
    if (meta['schema'] != SCHEMA or meta['base_assets_sha256'] != assets_sha256 or meta['rank'] != 8
            or meta['scaling'] != 2. or meta['implementation_sha256'] != sha(__file__)
            or meta['weights_sha256'] != sha(folder / 'adapter.safetensors')):
        raise ValueError('Changed or incompatible context adapter')
    tensors = load_file(str(folder / 'adapter.safetensors'))
    shapes = {n: list(p.shape) for n, p in adapters.items()}
    if set(tensors) != set(adapters) or meta['tensor_shapes'] != shapes or {n: list(p.shape) for n, p in tensors.items()} != shapes:
        raise ValueError('Context adapter tensor inventory differs')
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in tensors.values()):
        raise ValueError('Invalid context adapter tensor')
    with torch.no_grad():
        for n, p in adapters.items():
            p.copy_(tensors[n])
    return meta
