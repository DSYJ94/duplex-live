"""Native TRAIN-only sufficient statistics for a RET residual/bias comparison.

No parameter update or policy inference is performed here. Cache the actual
text-projection input, so later fitting cannot be mistaken for LM fine-tuning.
"""
import torch


def selected(features,logits,mask,targets):
    if features.ndim!=3 or features.shape[0]!=1 or logits.ndim!=4 or logits.shape[:3]!=(1,1,features.shape[1])or logits.shape[-1]<=4:
        raise ValueError('Actual one-sequence native projection input/output required')
    if mask.dtype!=torch.bool or mask.shape!=(features.shape[1],)or targets.shape!=mask.shape:raise ValueError('Exact original supervised positions required')
    x=features[0,mask];z=logits[0,0,mask]
    if not len(x)or not torch.isfinite(x).all()or not torch.isfinite(z).all():raise ValueError('Nonempty finite original TRAIN positions required')
    return dict(features=x.detach().cpu().contiguous(),RET=z[:,4].detach().cpu().contiguous(),other_logsumexp=torch.cat((z[:,:4],z[:,5:]),-1).float().logsumexp(-1).detach().cpu(),positive=(targets[mask]==4).detach().cpu())


def binary_logits(cache,delta):
    """Match BF16 native addition before the binary log-sum-exp objective."""
    x=cache['features'];ret=cache['RET'];other=cache['other_logsumexp']
    change=delta(x.float()).squeeze(-1).to(ret.dtype)
    return (ret+change).float()-other


def balanced_loss(logits,positive,positive_count,negative_count):
    from torch.nn import functional as F
    if logits.shape!=positive.shape or positive.dtype!=torch.bool or positive_count<1 or negative_count<1:raise ValueError('Declared complete two-class TRAIN population required')
    value=torch.where(positive,F.softplus(-logits)/positive_count,F.softplus(logits)/negative_count).sum()/2
    if not torch.isfinite(value):raise ValueError('Nonfinite native action objective')
    return value


def collect(lm,groups,guard):
    """All original TRAIN masks, preserving complete recorded prefix/cache calls."""
    from .matched_losses import RET_GROUPS,ret_population
    population=ret_population(groups);records=[];captured=[]
    if any(p.requires_grad for p in lm.parameters()):raise ValueError('Backbone and previous adapter must remain frozen')
    versions={n:p._version for n,p in lm.named_parameters()}
    def remember(module,args):captured.append(args[0].detach())
    hook=lm.text_linear.register_forward_pre_hook(remember)
    def forward(inputs,mask,targets):
        captured.clear();_,z=lm.forward_codes(inputs)
        if len(captured)!=1:raise ValueError('One actual native text projection per forward required')
        if not torch.isfinite(captured[0]).all()or not torch.isfinite(z).all():raise ValueError('Nonfinite recorded native prefix/output')
        return selected(captured[0],z,mask,targets)if mask.any()else None
    try:
        with torch.inference_mode():
            for group in sorted(RET_GROUPS):
                for row in groups[group]:
                    if row['fold']!='TRAIN':raise ValueError('No DEV/test action features may enter TRAIN')
                    guard(180);x=row['inputs'].cuda();mask=row['mask'].cuda();target=row['targets'].cuda();parts=[]
                    if group in ('stream49','control40','reasoning32'):
                        with lm.streaming(1):
                            for call in row['calls']:
                                guard(180);start,n=call['start'],call['frames'];part=forward(x[:,:,start:start+n],mask[start:start+n],target[start:start+n])
                                if part is not None:parts.append(part)
                    else:
                        part=forward(x,mask,target)
                        if part is not None:parts.append(part)
                    if parts:records.append(dict(group=group,id=row['id'],fold='TRAIN',**{k:torch.cat([p[k]for p in parts])for k in parts[0]}))
                    del x,mask,target
        pos=sum(int(r['positive'].sum())for r in records);neg=sum(len(r['positive'])-int(r['positive'].sum())for r in records)
        if (pos,neg)!=(population['positive'],population['negative']):raise ValueError('All original positive AND negative positions required')
        if any(p._version!=versions[n]or p.requires_grad for n,p in lm.named_parameters()):raise ValueError('Frozen native state changed during feature collection')
        return dict(schema='dl-native-ret-TRAIN-features-v1',population=population,rows=records,optimizer_updates=0,native_backbone_updated=False)
    finally:hook.remove()
