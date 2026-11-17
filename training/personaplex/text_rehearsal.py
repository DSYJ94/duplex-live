"""Explicit audio-absent causal language loss; corpus/GPU admission is separate.

Documents never become user/persona gold spans. Answer masking is an explicit
textual supervision convention, not a claim of an existing native chat API.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


def validate_window_identity(window, identity, admission, step):
    """Language exposure is a separately qualified, immutable schedule lineage."""
    import math
    binding=window.get('identity',{})
    keys={'corpus_sha256','tokenizer_sha256','schedule_sha256','coefficient'}
    if (set(binding)!=keys or binding!=identity.get('language_training')
            or type(window.get('step')) is not int or window['step']!=step
            or admission.get('language_data_qualified') is not True
            or admission.get('language_native_gradient_restore_passed') is not True):
        raise ValueError('Exact qualified language schedule and actual native restore required')
    for name in keys-{'coefficient'}:
        value=binding[name]
        if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Exact language corpus/tokenizer/schedule hashes required')
    coefficient=binding['coefficient']
    if isinstance(coefficient,bool) or not isinstance(coefficient,(int,float)) or not math.isfinite(coefficient) or coefficient<=0:
        raise ValueError('Explicit finite positive language mixture required')
    batches=window.get('batches',[])
    ids=window.get('document_ids',[])
    if (not batches or len(ids)!=len(batches) or len(ids)!=len(set(ids))
            or any(not isinstance(value,str) or not value for value in ids)):
        raise ValueError('Exact distinct local language document identities required')
    for batch in batches:
        validate(batch,window['spec'])


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


def backward_window(native_view, batches, spec, *, coefficient, group=None):
    """Accumulate one pooled-token language objective; caller owns step/zero_grad.

    Each rank must supply the same number of microbatches. Every backward uses
    the normal wrapped reducer; there is no additional accumulation division.
    Existing retained-objective gradients are deliberately left intact.
    """
    import math
    import torch.distributed as dist
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError('Explicit positive finite language mixture coefficient required')
    batches = list(batches)
    if not batches:
        raise ValueError('Complete nonempty language accumulation window required')
    validated = [validate(batch, spec) for batch in batches]
    device = validated[0][0].device
    if any(x.device != device for x, _, _ in validated):
        raise ValueError('One device per rank required')
    steps = torch.tensor(len(batches), device=device, dtype=torch.long)
    tokens = torch.tensor(sum(int(mask.sum()) for _, _, mask in validated), device=device, dtype=torch.long)
    if dist.is_initialized():
        low, high = steps.clone(), steps.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
        if int(low) != int(high):
            raise ValueError('Ranks must execute identical microbatch counts')
        dist.all_reduce(tokens, op=dist.ReduceOp.SUM, group=group)
    total = int(tokens)
    if total < 1:
        raise ValueError('Window has no qualified language targets')
    reports = []
    for batch in batches:
        value, detail = loss(native_view, batch, spec, group=group)
        weighted = value * (coefficient * detail['global_valid_text_tokens'] / total)
        weighted.backward()
        reports.append(dict(detail, weighted_loss=float(weighted.detach())))
    return dict(global_window_valid_text_tokens=total, microbatches=len(batches),
                coefficient=coefficient, optimizer_step_performed=False, batches=reports)


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
