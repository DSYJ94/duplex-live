"""Variance-reduced complete balanced microbatch; every approved term stays present."""
from collections import Counter
import json
import math
from pathlib import Path


def balanced_weights(kinds):
    if not kinds or any(not isinstance(k,str) or not k for k in kinds):raise ValueError('Verified nonempty kinds required')
    counts=Counter(kinds);return [1/(len(counts)*counts[k]) for k in kinds]


def validate_groups(groups):
    sizes={'actual':28,'reference':4,'old25':25,'new24':24,'word':10,'clock':14,'audio49':49}
    if set(groups)!=set(sizes):raise ValueError('Incomplete full balanced objective')
    for group,size in sizes.items():
        rows=groups[group];mass=64. if group in ('word','clock') else 1.
        if len(rows)!=size or len({x['id'] for x in rows})!=size or any(type(x['weight']) not in (int,float) or not math.isfinite(x['weight']) or x['weight']<=0 for x in rows) or abs(sum(x['weight'] for x in rows)-mass)>1e-12:
            raise ValueError('Changed full-batch coverage or objective mass: '+group)
        expected=balanced_weights([x['kind'] for x in rows]) if group in ('actual','old25','new24') else [mass/size]*size
        if any(abs(x['weight']-w)>1e-15 for x,w in zip(rows,expected)):raise ValueError('Changed balanced-kind or uniform weights')
    return True


def membership(data,parent):
    from evals.context_warm_ordinary_retention import IDS
    from evals.context_warm_ordinary_clock import CLOCK_IDS
    actual=data['original28_actual_components']
    groups={'actual':[dict(id=str(i)+':'+r['component'],kind=r.get('target_kind','private_result')) for i,r in enumerate(actual)],
        'reference':[dict(id=r['id']) for r in parent['retention_examples']],
        'word':[dict(id=i) for i in IDS],'clock':[dict(id=i) for i in CLOCK_IDS],
        'audio49':[dict(id=r['teacher']['id']) for r in data['cases']]}
    for group in ('old25','new24'):groups[group]=[dict(id=r['teacher']['id'],kind=r['teacher']['kind']) for r in data['cases'] if r['group']==group]
    for group,rows in groups.items():
        weights=balanced_weights([x['kind'] for x in rows]) if group in ('actual','old25','new24') else [(64. if group in ('word','clock') else 1.)/len(rows)]*len(rows)
        for row,w in zip(rows,weights):row['weight']=w
    validate_groups(groups);return groups


def matches(groups,expected):
    validate_groups(groups)
    for group,rows in expected.items():
        if [{k:r[k] for k in e} for r,e in zip(groups[group],rows)]!=rows:raise ValueError('Changed exact component membership: '+group)
    return True


def backward(lm,actual,retention,references,old,new,kinds,boundaries,audits,warm,clock,audio,forward,first_word_weight):
    import torch
    from torch.nn import functional as F
    from evals.context_decision_loss import weights
    from evals.context_teacher_text_balance import loss as teacher_loss
    from evals.context_warm_ordinary_retention import loss as anchor_loss
    from evals.context_native_audio49_training import forward as audio_forward
    from evals.context_native_audio49_loss import loss as audio_loss
    from evals.personaplex_adapters import checkpoint_language_layers
    groups={g:[] for g in ('actual','reference','old25','new24','word','clock','audio49')}
    def apply(group,identity,value,weight,**detail):
        if not torch.isfinite(value):raise RuntimeError('Nonfinite complete balanced objective')
        (value*weight).backward();groups[group].append(dict(id=identity,weight=weight,loss=float(value.detach()),**detail))
    with checkpoint_language_layers(lm):
        for i,((x,m,y),w) in enumerate(zip(actual,balanced_weights(kinds))):
            z=forward(x)[1][0,0,m].float();target=y[m];tw=weights(target,first_word_weight,boundaries[i])
            value=(F.cross_entropy(z,target,reduction='none')*tw).sum()/tw.sum()
            apply('actual',str(i)+':'+audits[i]['component'],value,w,scenario=audits[i]['scenario'],kind=kinds[i])
        for i,(x,m,record) in enumerate(retention):
            h,z=forward(x);bh,prob=references[i];kl=F.kl_div(z[0,0,m].float().log_softmax(-1),prob,reduction='batchmean');mse=(h[:,m].float()-bh.float()).square().mean()
            apply('reference',record['id'],kl+10*mse,1/4,kl=float(kl.detach()),hidden_mse=float(mse.detach()))
        for group,batches in [('old25',old),('new24',new)]:
            for (x,m,y,r),w in zip(batches,balanced_weights([b[3]['kind'] for b in batches])):
                value,detail=teacher_loss(forward(x)[1][0,0],y,m)
                apply(group,r['id'],value,w,kind=r['kind'],targets=int(m.sum()),objective=detail)
        for group,batches in [('word',warm),('clock',clock)]:
            for x,m,r,identity in batches:
                h,z=forward(x);value,kl,mse=anchor_loss(h,z,m,r)
                apply(group,identity,value,64/len(batches),kl=float(kl.detach()),hidden_mse=float(mse.detach()))
        for row in audio:
            codes=row['codes'].cuda();out=audio_forward(lm,codes);value,detail=audio_loss(out,codes,row['audio_mask'].cuda())
            apply('audio49',row['record']['teacher']['id'],value,1/49,objective=detail)
            del out,value,codes
    validate_groups(groups);return groups


def validate_exposure(steps,teacher_ids,unique_acoustic_targets,expected):
    if len(steps)!=8 or [x['step'] for x in steps]!=list(range(1,9)):raise ValueError('Complete batch is exactly one bounded8 stage')
    identity=None
    for step in steps:
        groups=step['complete_balanced_microbatches'];matches(groups,expected)
        current={g:[(x['id'],x['weight']) for x in rows] for g,rows in groups.items()}
        if identity is None:identity=current
        if current!=identity:raise ValueError('Full balanced membership or weights changed within stage')
        audio=groups['audio49']
        if [x['id'] for x in audio]!=teacher_ids or any(x['weight']!=1/49 for x in audio):raise ValueError('Missing approved acoustic teacher')
        count=0
        for row in audio:
            d=row['objective'];values=d['assistant_audio_targets']
            if d['user_audio_loss_targets']!=0 or len(values)!=8 or any(type(v) is not int or v<1 for v in values):raise ValueError('Changed audio ownership')
            count+=sum(values)
        if count!=unique_acoustic_targets:raise ValueError('Changed complete49 acoustic target count')
        if step['assistant_audio49_microbatch']!=audio or step['additional_native_v2_microbatch']!=groups['new24']:raise ValueError('Conflicting step exposure receipts')
    return True


def preflight(lm,learned,args,policy,*inputs):
    import torch
    from evals.moshi_training_smoke import tensor_hashes
    from evals.moshi_model_download import sha256
    before=tensor_hashes(lm.named_parameters())
    for p in learned.values():p.grad=None
    groups=backward(lm,*inputs)
    matches(groups,policy['complete_batch_membership'])
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in learned.values()):raise ValueError('Missing/nonfinite complete gradient')
    if any(p.grad is not None for n,p in lm.named_parameters() if n not in learned):raise ValueError('Frozen parameter received a gradient')
    norm=sum(float(p.grad.detach().double().square().sum()) for p in learned.values())**.5
    if norm<=0 or before!=tensor_hashes(lm.named_parameters()):raise ValueError('Complete gradient dead or parameters changed')
    for p in learned.values():p.grad=None
    peak=torch.cuda.max_memory_allocated()
    if peak>=80_000_000_000:raise ValueError('Complete-batch readiness exceeded80GB')
    report=dict(schema='dl-native-complete-balanced-preflight-v1',passed=True,policy=policy,groups=groups,total_components=154,combined_gradient_norm=norm,peak_allocated_bytes=peak,
        all_parameters_unchanged=True,all_trainable_gradients_finite=True,all_trainable_gradients_present=True,frozen_parameter_gradients=0,
        optimizer_updates=0,new_checkpoint_written=False,training_approved=False,source_commit=__import__('os').environ['DL_TASK_COMMIT'])
    (args.out/'complete-balanced-preflight.json').write_text(json.dumps(report,indent=2)+'\n');return sha256(args.out/'complete-balanced-preflight.json')
