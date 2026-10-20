"""Add assistant acoustic CE beside every original text/retention objective.

Callers must independently approve complete49 code ownership and native depth
numerics. Outputs already undo native codec delays; never shift labels here.
"""

def loss(output,codes,audio_mask):
    import torch
    from torch.nn import functional as F
    if (codes.dtype!=torch.int64 or codes.ndim!=3 or codes.shape[1]!=17
            or audio_mask.dtype!=torch.bool or audio_mask.shape!=(codes.shape[0],codes.shape[2])
            or output.logits.shape[:3]!=(codes.shape[0],16,codes.shape[2])
            or output.mask.shape!=(codes.shape[0],16,codes.shape[2]) or output.mask.dtype!=torch.bool):
        raise ValueError('Invalid aligned assistant-audio supervision')
    keep=output.mask[:,:8]&audio_mask[:,None,:];parts=[]
    for q in range(8):
        mask=keep[:,q];z=output.logits[:,q][mask].float();y=codes[:,q+1][mask]
        if not len(y) or not torch.isfinite(z).all() or (y<0).any() or (y>=z.shape[-1]).any():
            raise ValueError('Missing or invalid approved assistant acoustic targets')
        parts.append(F.cross_entropy(z,y))
    value=(parts[0]+.02*sum(parts[1:]))/1.14
    if not torch.isfinite(value):raise ValueError('Nonfinite assistant audio loss')
    return value,dict(assistant_audio_targets=[int(x) for x in keep.sum((0,2))],
        per_head_ce=[float(x.detach()) for x in parts],user_audio_loss_targets=0,
        output_already_aligned=True,prefix_and_span_exclusion_requires_approved_audio_mask=True,
        original_text_and_kl_hidden_losses_must_be_preserved=True)
