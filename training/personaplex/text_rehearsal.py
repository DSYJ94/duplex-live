"""Explicit audio-absent causal language loss; corpus/GPU admission is separate.

Documents never become user/persona gold spans. Answer masking is an explicit
textual supervision convention, not a claim of an existing native chat API.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TextSpec:
    audio_streams: int
    vocabulary: int
    initial_token: int
    padding_token: int
    max_tokens: int
    reserved_tokens: frozenset

    def __post_init__(self):
        values = (self.audio_streams, self.vocabulary, self.initial_token, self.padding_token, self.max_tokens)
        if (any(type(v) is not int for v in values) or self.audio_streams < 1
                or self.vocabulary <= 4 or self.max_tokens < 1
                or not 0 <= self.initial_token < self.vocabulary
                or not 0 <= self.padding_token < self.vocabulary
                or not isinstance(self.reserved_tokens, frozenset)
                or not {self.initial_token, self.padding_token, 4} <= self.reserved_tokens
                or any(type(v) is not int or not 0 <= v < self.vocabulary for v in self.reserved_tokens)):
            raise ValueError('Exact native geometry/BOS/padding/reserved-control declaration required')


def pack(document_tokens, spec, *, supervise=None, device='cpu'):
    ids = list(document_tokens)
    if (not ids or len(ids) > spec.max_tokens
            or any(type(v) is not int or not 0 <= v < spec.vocabulary or v in spec.reserved_tokens for v in ids)):
        raise ValueError('Complete bounded ordinary-token document required; no silent truncation or control targets')
    active = [True] * len(ids) if supervise is None else list(supervise)
    if len(active) != len(ids) or any(type(v) is not bool for v in active) or not any(active):
        raise ValueError('Explicit nonempty qualified target mask required')
    targets = torch.tensor([ids], dtype=torch.long, device=device)
    codes = torch.full((1, spec.audio_streams + 1, len(ids)), -1, dtype=torch.long, device=device)
    codes[0, 0] = torch.tensor([spec.initial_token] + ids[:-1], dtype=torch.long, device=device)
    return dict(inputs=codes, targets=targets,
                mask=torch.tensor([active], dtype=torch.bool, device=device))


def validate(batch, spec):
    x, y, mask = (batch[k] for k in ('inputs', 'targets', 'mask'))
    if (x.ndim != 3 or x.shape[1] != spec.audio_streams + 1
            or not 1 <= x.shape[-1] <= spec.max_tokens or x.shape[0] < 1
            or x.dtype != torch.long or y.dtype != torch.long or mask.dtype != torch.bool
            or y.shape != (x.shape[0], x.shape[-1]) or mask.shape != y.shape
            or x.device != y.device or x.device != mask.device
            or bool((x[:, 1:] != -1).any()) or bool((x[:, 0, 0] != spec.initial_token).any())
            or not torch.equal(x[:, 0, 1:], y[:, :-1])
            or bool(((y < 0) | (y >= spec.vocabulary)).any())
            or any(bool((y == value).any()) for value in spec.reserved_tokens)):
        raise ValueError('Exact one-token causal shift and absent audio must remain intact')
    return x, y, mask


def averaged_gradient_loss(local_sum, global_tokens, world_size):
    """DDP/FSDP average gradients: world/global count yields a pooled token mean."""
    if type(global_tokens) is not int or global_tokens < 1 or type(world_size) is not int or world_size < 1:
        raise ValueError('Positive global token population and real averaging world required')
    return local_sum * (world_size / global_tokens)


def loss(native_view, batch, spec, *, group=None):
    """Return differentiable globally token-weighted CE; no optimizer/backward here."""
    import torch.distributed as dist
    x, y, mask = validate(batch, spec)
    if (native_view.n_q != spec.audio_streams
            or native_view.text_initial_token_id != spec.initial_token
            or native_view.text_linear.out_features != spec.vocabulary):
        raise ValueError('Declared text spec differs from actual native LM')
    hidden, logits = native_view.forward_text_rehearsal(x)
    if (logits.shape != (x.shape[0], 1, x.shape[-1], spec.vocabulary)
            or not torch.isfinite(hidden).all() or not torch.isfinite(logits).all()):
        raise ValueError('Complete finite temporal/text output required')
    local_tokens = int(mask.sum())
    count = torch.tensor(local_tokens, dtype=torch.long, device=logits.device)
    world = 1
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=group)
        world = dist.get_world_size(group)
    total = int(count)
    if total < 1:
        raise ValueError('No qualified text targets in global batch')
    selected = logits[:, 0][mask].float()
    local_sum = F.cross_entropy(selected, y[mask], reduction='sum') if local_tokens else logits.sum() * 0.
    value = averaged_gradient_loss(local_sum, total, world)
    if not torch.isfinite(value):
        raise ValueError('Nonfinite text rehearsal loss')
    return value, dict(local_valid_text_tokens=local_tokens, global_valid_text_tokens=total,
                       gradient_averaging_world_size=world, local_CE_sum=float(local_sum.detach()),
                       optimizer_step_performed=False, acoustic_loss_computed=False,
                       objective='audio_absent_next_token_language', native_speech_transfer_verified=False)
