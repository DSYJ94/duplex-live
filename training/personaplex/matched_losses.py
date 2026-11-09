"""Preserved native objectives plus separately normalized TRAIN-only RET loss.

No optimizer ownership, sampling, data selection or acceptance decision lives
here. The caller supplies the complete admitted corpus and scheduled exposure.
"""
from pathlib import Path
import math

from evals.moshi_model_download import sha256 as sha
from training.personaplex.matched_schedule import STEPS

RET_GROUPS=frozenset(('actual','stream49','control40','reasoning32','ordinary48'))


def ret_population(groups):
    """Original supervised positions only; prefixes/spans remain excluded.

    The native49 teachers enter once, through stream49 rather than again through
    old25/new24. Actual-history components retain their distinct conditioning.
    Positive and negative class mass each sum to half the proposed coefficient8.
    """
    positives=negatives=0
    for group in RET_GROUPS:
        for row in groups[group]:
            if row['fold']!='TRAIN':raise ValueError('No evaluation target in RET auxiliary')
            labels=row['targets'][row['mask']]
            positives+=int((labels==4).sum());negatives+=int((labels!=4).sum())
    if positives<1 or negatives<1:raise ValueError('Both original RET decision classes required')
    return dict(positive=positives,negative=negatives,coefficient=8.,
        positive_position_weight=4./positives,negative_position_weight=4./negatives,
        groups=sorted(RET_GROUPS),new_targets=0)


def ret_loss(logits,labels,population):
    """Stable binary RET-vs-all-other-token CE, including extreme wrong logits."""
    import torch
    from torch.nn import functional as F
    if logits.ndim!=2 or logits.shape[1]<=4 or labels.shape!=(logits.shape[0],):
        raise ValueError('Selected original native text positions required')
    other=torch.cat([logits[:,:4],logits[:,5:]],dim=-1).logsumexp(-1)
    binary=logits[:,4]-other
    positive=labels==4
    values=torch.where(positive,F.softplus(-binary)*population['positive_position_weight'],
        F.softplus(binary)*population['negative_position_weight'])
    value=values.sum()
    if not torch.isfinite(value):raise ValueError('Nonfinite native RET auxiliary')
    return value


def recorded_text(lm,inputs,mask,targets,calls,weight,*,population=None,ret_scale=0.):
    """Same recorded-call CE/cache-detach semantics as the retained native helper.

    This integrates the supplemental RET term without a second streaming replay.
    Cache VALUES stay unchanged; autograd is truncated at recorded boundaries.
    """
    import torch
    from torch.nn import functional as F
    from evals.context_native_stream_gradient import coefficients,detach_caches
    from evals.context_native_bf16_stream_diagnostic import call_partition
    if not math.isfinite(weight) or weight<=0 or not math.isfinite(ret_scale) or ret_scale<0:
        raise ValueError('Declared finite objective weights required')
    if bool(population)!=(ret_scale>0):raise ValueError('RET population and selection weight must agree')
    call_partition(calls,inputs.shape[-1]);coef=coefficients(targets,mask)
    total=extra=0.;gradient_calls=0
    with lm.streaming(1):
        for call in calls:
            start,n=call['start'],call['frames'];active=mask[start:start+n]
            if call.get('phase') in ('prefix','span') and active.any():
                raise ValueError('Prefix or provided span acquired a target')
            h,z=lm.forward_codes(inputs[:,:,start:start+n])
            if not torch.isfinite(h).all() or not torch.isfinite(z).all():raise ValueError('Nonfinite recorded native output')
            if active.any():
                labels=targets[start:start+n][active];logits=z[0,0,active].float()
                ce=(F.cross_entropy(logits,labels,reduction='none')*coef[start:start+n][active]).sum()
                auxiliary=ret_loss(logits,labels,population) if population else ce.new_zeros(())
                value=ce*weight+auxiliary*ret_scale
                if not torch.isfinite(value):raise ValueError('Nonfinite recorded native loss')
                value.backward();gradient_calls+=1
                total+=float(ce.detach());extra+=float(auxiliary.detach())
                del value,ce,auxiliary,logits
            detach_caches(lm)
            del h,z
    return dict(text_ce=total,ret_raw=extra,weighted_loss=weight*total+ret_scale*extra,
        gradient_calls=gradient_calls,recorded_call_count=len(calls),
        native_text_truncation='recorded call boundaries; forward KV values unchanged')


class Objective:
    def __init__(self,lm,groups,references):
        self.lm,self.groups,self.references=lm,groups,references
        self.ret=ret_population(groups);self.anchors={}
        if set(references)!={r['id'] for r in groups['reference']}:
            raise ValueError('Every original four-way frozen reference required')

    def anchor(self,row,device):
        import torch
        path=row['anchor_file']
        if path not in self.anchors:
            if sha(Path(path))!=row['anchor_sha256']:raise ValueError('Protected native520 anchor changed')
            self.anchors[path]=torch.load(path,weights_only=True,map_location='cpu')
        values=self.anchors[path];prefix=row['anchor_tensor_prefix']
        return values[prefix+'.hidden'].to(device),values[prefix+'.logp'].to(device)

    def backward(self,exposure):
        import torch
        from torch.nn import functional as F
        from evals.personaplex_adapters import checkpoint_language_layers
        from evals.context_decision_loss import weights
        from evals.context_teacher_text_balance import loss as teacher_loss
        from evals.context_native_stream_gradient import coefficients
        from evals.context_warm_ordinary_retention import loss as anchor_loss
        from evals.context_native_audio49_training import forward as audio_forward
        from evals.context_native_audio49_loss import loss as audio_loss
        from evals.native_control_numerical import target_positions
        group=exposure['group'];row=self.groups[group][exposure['index']];w=exposure['weight']
        if row['id']!=exposure['id'] or row['fold']!='TRAIN':raise ValueError('Scheduled original TRAIN identity differs')
        ret_scale=STEPS/exposure['visits_this_epoch'] if group in RET_GROUPS else 0.
        result=dict(group=group,id=row['id'],weight=w,ret_scale=ret_scale,weighted_loss=0.)
        if group in ('stream49','control40','reasoning32'):
            result.update(recorded_text(self.lm,row['inputs'].cuda(),row['mask'].cuda(),row['targets'].cuda(),
                row['calls'],w,population=self.ret,ret_scale=ret_scale))
        elif group!='audio49':
            x,m=row['inputs'].cuda(),row['mask'].cuda()
            with checkpoint_language_layers(self.lm):
                h,z=self.lm.forward_codes(x)
                auxiliary=z.new_zeros((),dtype=torch.float32)
                if group=='reference':
                    bh,prob=self.references[row['id']]
                    kl=F.kl_div(z[0,0,m].float().log_softmax(-1),prob.to(z.device),reduction='batchmean')
                    mse=(h[:,m].float()-bh.to(h.device).float()).square().mean();value=kl+10.*mse
                    result.update(kl=float(kl.detach()),hidden_mse=float(mse.detach()))
                elif group in ('word','clock'):
                    value,kl,mse=anchor_loss(h,z,m,self.anchor(row,z.device))
                    result.update(kl=float(kl.detach()),hidden_mse=float(mse.detach()))
                else:
                    y=row['targets'].cuda();labels=y[m];logits=z[0,0,m].float()
                    if group=='actual':
                        tw=weights(labels,16.,row['boundary']);value=(F.cross_entropy(logits,labels,reduction='none')*tw).sum()/tw.sum()
                    elif group in ('old25','new24'):
                        value,_=teacher_loss(z[0,0],y,m)
                    elif group=='ordinary48':
                        value=(F.cross_entropy(logits,labels,reduction='none')*coefficients(y,m)[m]).sum()
                    else:raise ValueError('Undeclared objective group')
                    result['text_ce']=float(value.detach())
                    if ret_scale:auxiliary=ret_loss(logits,labels,self.ret)
                combined=w*value+ret_scale*auxiliary
                if not torch.isfinite(combined):raise ValueError('Nonfinite retained native objective')
                combined.backward();result['weighted_loss']=float(combined.detach())
                result['ret_raw']=float(auxiliary.detach())
            del x,m,h,z,value,combined,auxiliary
        if group in ('audio49','control40','reasoning32','ordinary48'):
            codes,mask=row['codes'].cuda(),row['audio_mask'].cuda()
            with checkpoint_language_layers(self.lm):
                output=audio_forward(self.lm,codes)
                value,detail=audio_loss(output,codes,mask)
                _,counts=target_positions(mask[0],codes,output)
                if detail['assistant_audio_targets']!=counts:raise ValueError('Complete eight-head assistant ownership changed')
                (w*value).backward();result['weighted_loss']+=w*float(value.detach())
                result.update(acoustic_ce=float(value.detach()),acoustic_objective=detail)
            del codes,mask,output,value
        return result
