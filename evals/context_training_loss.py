"""Native aligned 16-head loss with explicit speaker-target ownership.

User targets come from clean user stems. Never suppress genuine user overlap
because the assistant is speaking. Span/prefix and invalid labels are excluded
before CE. This mask interface is not a waveform echo canceller.
"""


def loss_fn(out, codes, text_mask, audio_mask, user_target_mask, lm):
    import torch
    import torch.nn.functional as F
    if lm.audio_offset != 1 or out.logits.shape[1] != 16:
        raise RuntimeError('Expected native assistant-eight/user-eight objective')
    if any(m.dtype != torch.bool or m.shape != codes[:, 0].shape
           for m in (text_mask, audio_mask, user_target_mask)):
        raise RuntimeError('Expected aligned boolean target masks')
    if (user_target_mask & ~audio_mask).any():
        raise RuntimeError('User ownership mask cannot enable virtual/prefix targets')
    tl = out.text_logits[:, 0]
    tgt = codes[:, 0]
    keep = out.text_mask[:, 0] & text_mask & (tgt >= 0)
    if not keep.any() or not torch.isfinite(tl[keep]).all():
        raise RuntimeError('Absent or nonfinite valid text predictions')
    if (tgt[keep] >= tl.shape[-1]).any():
        raise RuntimeError('Text target exceeds native vocabulary')
    weights = torch.where(tgt[keep] == 3, .5, 1.)
    text_loss = (F.cross_entropy(tl[keep].float(), tgt[keep], reduction='none') * weights).sum() / weights.sum()
    al = out.logits
    at = codes[:, 1:17]
    valid = out.mask & audio_mask[:, None, :] & (at >= 0)
    valid[:, 8:] &= user_target_mask[:, None, :]
    if not valid.any() or not torch.isfinite(al[valid]).all():
        raise RuntimeError('Absent or nonfinite valid audio predictions')
    if (at[valid] >= al.shape[-1]).any():
        raise RuntimeError('Audio target exceeds native vocabulary')
    per_head = valid.sum((0, 2))
    if not (per_head > 0).all():
        raise RuntimeError('A required native audio head has no eligible targets')
    cbw = torch.full((16,), .02, device=codes.device)
    cbw[0] = 1.
    aw = cbw[None, :, None].expand_as(at)[valid]
    ce = F.cross_entropy(al[valid].float(), at[valid], reduction='none')
    audio_loss = (ce * aw).sum() / aw.sum()
    counts = {'text': int(keep.sum()), 'assistant_audio': int(valid[:, :8].sum()),
        'clean_user_audio': int(valid[:, 8:].sum()),
        'user_targets_excluded_by_ownership': int((out.mask[:, 8:] & audio_mask[:, None, :]
            & (at[:, 8:] >= 0) & ~user_target_mask[:, None, :]).sum()),
        'per_audio_head': per_head.tolist()}
    return text_loss + audio_loss, {'text_loss': float(text_loss.detach()),
        'audio_loss': float(audio_loss.detach()), 'target_counts': counts}
