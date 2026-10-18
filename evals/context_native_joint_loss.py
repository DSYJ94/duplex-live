"""Assistant-only native text/audio loss on already aligned forward_train output.

User audio remains conditioning. None of its eight codebooks is a target, so
assistant echo present on that stream cannot become user prediction supervision.
This is loss ownership, not waveform echo cancellation. The upstream forward
already reverses codec delays: never shift its logits or labels again here.
Callers must independently approve the real-column masks and source recordings.
"""


def loss(output, codes, text_mask, audio_mask, *, audio_weight=1.):
    import math
    import torch
    from torch.nn import functional as F

    if isinstance(audio_weight, bool) or not math.isfinite(audio_weight) or audio_weight < 0:
        raise ValueError('Audio loss weight must be finite and nonnegative')
    if codes.dtype != torch.int64 or codes.ndim != 3 or codes.shape[1] != 17:
        raise ValueError('Expected clean text/assistant-eight/user-eight targets')
    batch, _, frames = codes.shape
    shape = (batch, frames)
    if any(x.dtype != torch.bool or x.shape != shape for x in (text_mask, audio_mask)):
        raise ValueError('Expected independently approved real-column masks')
    tl, al = output.text_logits, output.logits
    if (tl.ndim != 4 or tl.shape[:3] != (batch, 1, frames)
            or al.ndim != 4 or al.shape[:3] != (batch, 16, frames)
            or output.text_mask.shape != (batch, 1, frames)
            or output.mask.shape != (batch, 16, frames)
            or output.text_mask.dtype != torch.bool or output.mask.dtype != torch.bool):
        raise ValueError('Unexpected pinned native output layout')
    keep_text = output.text_mask[:, 0] & text_mask
    keep_audio = output.mask[:, :8] & audio_mask[:, None, :]
    if not keep_text.any() or not keep_audio.any():
        raise ValueError('Joint supervision requires text and assistant audio')

    def checked_ce(logits, targets, keep):
        z, y = logits[keep].float(), targets[keep]
        if not torch.isfinite(z).all() or (y < 0).any() or (y >= z.shape[-1]).any():
            raise ValueError('Invalid supervised native logits or targets')
        return F.cross_entropy(z, y, reduction='none'), y

    text_ce, y = checked_ce(tl[:, 0], codes[:, 0], keep_text)
    spoken = y >= 5
    clock = ~spoken
    pieces = []
    if spoken.any():
        pieces.append(text_ce[spoken].mean())
    if clock.any():
        weights = torch.where(y[clock] == 3, .5, 1.)
        pieces.append((text_ce[clock] * weights).sum() / weights.sum())
    # Pure silence is valid supervision too; no obligatory spoken reply.
    text_loss = sum(pieces) / len(pieces)
    acoustic = []
    for q in range(8):
        keep = keep_audio[:, q]
        if not keep.any():
            raise ValueError('An assistant codebook has no valid targets')
        ce, _ = checked_ce(al[:, q], codes[:, 1 + q], keep)
        acoustic.append(ce.mean())
    # Same semantic/acoustic ratio as the pinned upstream fine-tuner.
    audio_loss = (acoustic[0] + .02 * sum(acoustic[1:])) / 1.14
    total = text_loss + audio_weight * audio_loss
    if not torch.isfinite(total):
        raise ValueError('Nonfinite native joint objective')
    return total, dict(text_ce=float(text_loss.detach()), audio_ce=float(audio_loss.detach()),
        audio_weight=audio_weight, text_targets=int(keep_text.sum()),
        spoken_targets=int(spoken.sum()), clock_targets=int(clock.sum()),
        assistant_audio_targets=[int(x) for x in keep_audio.sum(dim=(0, 2))],
        user_audio_loss_targets=0, aligned_output_not_shifted=True,
        excludes_columns_outside_approved_masks=True, waveform_aec=False)
