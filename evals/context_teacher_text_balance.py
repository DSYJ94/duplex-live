"""Explicit spoken/clock balancing on already approved native teacher text.

No targets are added or removed. Each group gets half the objective; PAD3
retains weight 0.5 within the clock/control group. This is an experiment,
not evidence that target imbalance explains free-running failures.
"""


def validate(mode, native, coverage):
    if mode not in ('legacy', 'balanced-spoken'):
        raise ValueError('Unknown native teacher text loss')
    if mode != 'legacy' and not (native and coverage == 'all-approved'):
        raise ValueError('Balanced spoken loss requires complete approved native teachers')


def loss(logits, labels, mask):
    import torch
    from torch.nn import functional as F
    if (logits.ndim != 2 or labels.dtype != torch.int64 or mask.dtype != torch.bool
            or labels.shape != mask.shape or labels.shape != (logits.shape[0],)
            or not mask.any()):
        raise RuntimeError('Invalid native teacher text coverage')
    z = logits[mask].float(); y = labels[mask]
    if not torch.isfinite(z).all() or (y < 0).any() or (y >= z.shape[1]).any():
        raise RuntimeError('Invalid native teacher text logits or labels')
    spoken = y >= 5
    if not spoken.any() or spoken.all():
        raise RuntimeError('Balanced teacher requires both measured spoken and clock/control targets')
    ce = F.cross_entropy(z, y, reduction='none')
    weights = torch.where(y == 3, .5, 1.)
    speech = ce[spoken].mean()
    clock = (ce[~spoken] * weights[~spoken]).sum() / weights[~spoken].sum()
    result = .5 * (speech + clock)
    diagnostic = dict(spoken_targets=int(spoken.sum()), clock_control_targets=int((~spoken).sum()),
        spoken_ce=float(speech.detach()), clock_control_ce=float(clock.detach()),
        legacy_spoken_weight_mass=float(weights[spoken].sum() / weights.sum()),
        balanced_spoken_weight_mass=.5, balanced_clock_control_weight_mass=.5,
        prefix_loss_targets=0, span_loss_targets=0, user_loss_targets=0, audio_loss_targets=0)
    return result, diagnostic
