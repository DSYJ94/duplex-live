"""Native text/acoustic SFT for admitted new pairs; equal-example global mass.

Retained history/KL/hidden objectives remain separate and unchanged. Each rank
executes one text forward/backward and one complete native acoustic forward/
backward, including zero-weight padding ranks. No RET targets are fabricated.
"""
import math
import torch
import torch.distributed as dist


def scale(valid, coefficient, global_batch_count=None):
    if type(valid) is not bool or not math.isfinite(coefficient) or coefficient<=0:
        raise ValueError('Declared positive new-data objective mass required')
    local=dict(valid=valid,coefficient=coefficient,global_batch_count=global_batch_count)
    flags=[None]*dist.get_world_size();dist.all_gather_object(flags,local)
    if any(x['coefficient']!=coefficient or x['global_batch_count']!=global_batch_count for x in flags):raise ValueError('Ranks disagree on complete global batch weighting')
    count=sum(x['valid'] for x in flags)
    if not count:raise ValueError('Empty global new-data microbatch')
    denominator=count if global_batch_count is None else global_batch_count
    if type(denominator) is not int or denominator<count:raise ValueError('Whole global batch count required')
    return coefficient*dist.get_world_size()/denominator if valid else 0.


def backward(runtime,row,*,valid,coefficient,global_batch_count=None):
    from evals.personaplex_adapters import checkpoint_language_layers
    from evals.context_native_stream_gradient import coefficients
    from evals.context_native_audio49_training import forward as audio_forward
    from evals.context_native_audio49_loss import loss as audio_loss
    from evals.native_control_numerical import target_positions
    from torch.nn import functional as F
    required=('id','fold','inputs','mask','targets','codes','audio_mask','native_qualification')
    if any(k not in row for k in required) or row['fold']!='TRAIN' or row['native_qualification'] is not True:
        raise ValueError('Complete independently admitted native TRAIN pair required')
    mass=scale(valid,coefficient,global_batch_count);lm=runtime.view;device=runtime.device
    x,m,y=(row[k].to(device) for k in ('inputs','mask','targets'))
    if not m.any():raise ValueError('Real example required even for zero-mass padding')
    with checkpoint_language_layers(lm):
        h,z=lm.forward_codes(x)
        if not torch.isfinite(h).all() or not torch.isfinite(z).all():raise ValueError('Nonfinite native text output')
        text=(F.cross_entropy(z[0,0,m].float(),y[m],reduction='none')*coefficients(y,m)[m]).sum()
        if not torch.isfinite(text):raise ValueError('Nonfinite expanded text loss')
        (text*mass).backward();text_value=float(text.detach())
    del h,z,text,x,m,y
    codes,mask=row['codes'].to(device),row['audio_mask'].to(device)
    with checkpoint_language_layers(lm):
        output=audio_forward(lm,codes);audio,detail=audio_loss(output,codes,mask)
        _,counts=target_positions(mask[0],codes,output)
        if detail['assistant_audio_targets']!=counts:raise ValueError('Complete native assistant audio ownership required')
        if not torch.isfinite(audio):raise ValueError('Nonfinite expanded acoustic loss')
        (audio*mass).backward();audio_value=float(audio.detach())
    return dict(id=row['id'],valid=valid,global_example_scale=mass,text_ce=text_value,
        acoustic_ce=audio_value,weighted_loss=mass*(text_value+audio_value),
        acoustic_objective=detail,new_RET_targets=0)
