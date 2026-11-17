"""Fixed-recipe supervised controls on actual frozen native TRAIN features.

This is not LM fine-tuning, native-policy acceptance, RL or autonomous OPAL.
Full upstream OPAL/Adam/RNG state is preserved; future code changes use the
existing local-Git broker only after a backed native behavioral boundary.
"""
import json,os
from pathlib import Path
import torch
from torch import nn
from .action_features import binary_logits,balanced_loss

RECIPE=dict(updates=128,warmup_updates=16,learning_rate=.001,gradient_clip=1.,weight_decay=0.,adam_betas=(.9,.999),adam_eps=1e-8)


def model(features,mode,device):
    if mode not in ('readout','bias')or type(features)is not int or features<1:raise ValueError('Explicit native readout/bias control required')
    result=nn.Linear(features,1,bias=True,device=device,dtype=torch.float32)
    nn.init.zeros_(result.weight);nn.init.zeros_(result.bias)
    if mode=='bias':result.weight.requires_grad_(False)
    return result


def validate(cache):
    if cache.get('schema')!='dl-native-ret-TRAIN-features-v1'or cache.get('optimizer_updates')!=0 or cache.get('native_backbone_updated')is not False:raise ValueError('Actual zero-update frozen native TRAIN cache required')
    rows=cache['rows'];pop=cache['population'];keys=set();positive=negative=0;features=None
    for row in rows:
        key=(row['group'],row['id'])
        if key in keys or row['fold']!='TRAIN':raise ValueError('Distinct original TRAIN components only')
        keys.add(key);x=row['features']
        if not torch.is_tensor(x)or x.ndim!=2:raise ValueError('Actual two-dimensional native feature matrix required')
        n=len(x);d=x.shape[-1]
        if x.ndim!=2 or n<1 or row['RET'].shape!=(n,)or row['other_logsumexp'].shape!=(n,)or row['positive'].shape!=(n,)or row['positive'].dtype!=torch.bool:raise ValueError('Complete native action feature shapes required')
        if features is not None and d!=features:raise ValueError('One native projection width required')
        if any(not torch.isfinite(row[k]).all()for k in ('features','RET','other_logsumexp')):raise ValueError('Finite actual native features required')
        features=d;positive+=int(row['positive'].sum());negative+=n-int(row['positive'].sum())
    if features is None or (positive,negative)!=(pop['positive'],pop['negative'])or positive<1 or negative<1:raise ValueError('Whole positive AND negative population required')
    return features


def evaluate(delta,rows):
    p=[];n=[]
    with torch.no_grad():
        for row in rows:
            probability=binary_logits(row,delta).sigmoid();p.append(probability[row['positive']]);n.append(probability[~row['positive']])
        a=torch.cat(p);b=torch.cat(n)
    return dict(positive_count=len(a),negative_count=len(b),positive_mean_probability=float(a.mean()),positive_probability_above_half=int((a>.5).sum()),negative_mean_probability=float(b.mean()),negative_probability_above_half=int((b>.5).sum()),teacher_forced_TRAIN_only=True,native_free_running_qualified=False)


def new_runtime(delta,mode,*,opal_source,out):
    from training.venus.native_optimizer import NativeOpal
    from .matched_state import materialize_adam
    if mode not in ('readout','bias'):raise ValueError('Explicit paired action mode required')
    delta.weight.requires_grad_(mode=='readout');delta.bias.requires_grad_(True)
    recipe=dict(RECIPE,learning_rate=.5 if mode=='bias' else RECIPE['learning_rate'])
    base=torch.optim.AdamW([p for p in delta.parameters()if p.requires_grad],lr=recipe['learning_rate'],betas=RECIPE['adam_betas'],eps=RECIPE['adam_eps'],weight_decay=0.,foreach=False,fused=False,amsgrad=False)
    materialize_adam(base)
    runtime=NativeOpal(base,delta,source=opal_source,run_dir=out,recipe=recipe,backend='keep',goal='Fixed supervised RET separability control; functional native gates determine usefulness')
    return runtime,recipe


def fit(cache,out,*,mode,device,opal_source,guard=lambda needed=0:None,identity,provenance=None):
    from training.venus.opal_adapter import checkpoint_manifest
    from training.venus.state import rng_state
    from .distributed_native import cpu_state
    features=validate(cache);out=Path(out);out.mkdir(exist_ok=False);delta=model(features,mode,device)
    # A scalar offset needs an order-one shift, unlike a4096-coordinate head.
    # An artificially tiny bias LR would be an underpowered capacity control.
    rows=[dict(r,**{k:r[k].to(device)for k in ('features','RET','other_logsumexp','positive')})for r in cache['rows']]
    opal,recipe=new_runtime(delta,mode,opal_source=opal_source,out=out/'opal');base=opal.base
    metadata=dict(mode=mode,input_features=features,trainable_parameters=sum(p.numel()for p in delta.parameters()if p.requires_grad),identity=identity,recipe=recipe,backbone_updates=0,previous_saved_updates_replayed=0,OPAL_mode='KEEP_FIXED_CONTROL_NOT_AUTONOMOUS_REWRITING',provenance=provenance or {})
    def save(step):
        manifest=checkpoint_manifest(opal.optimizer)
        if manifest['update']!=step:raise ValueError('Exactly the actual new readout update required')
        metrics=evaluate(delta,rows);state=dict(schema='dl-native-ret-fit-state-v1',metadata=metadata,step=step,writer_pid=os.getpid(),delta=cpu_state(delta.state_dict()),optimizer=cpu_state(opal.optimizer.state_dict()),rng=rng_state(),diagnostics=opal.diagnostics_state,metrics=metrics,program=manifest)
        path=out/('step-%04d.pt'%step);temporary=path.with_suffix('.tmp');torch.save(state,temporary);os.replace(temporary,path);return metrics,path
    try:
        initial,_=save(0)
        with (out/'training.jsonl').open('x')as log:
            for step in range(RECIPE['updates']):
                guard(180);base.zero_grad(set_to_none=True);total=0.
                for row in rows:
                    loss=balanced_loss(binary_logits(row,delta),row['positive'],cache['population']['positive'],cache['population']['negative']);loss.backward();total+=float(loss.detach())
                if any(p.grad is None or not torch.isfinite(p.grad).all()for p in delta.parameters()if p.requires_grad):raise ValueError('Complete finite readout gradients required')
                program=opal.step(total,step);log.write(json.dumps(dict(actual_new_readout_step=step+1,loss=total,program=program,backbone_updates=0))+'\n');log.flush()
                if (step+1)%16==0:metrics,path=save(step+1)
        result=dict(metadata,actual_new_readout_updates=128,initial=initial,final=metrics,checkpoint=str(path),native_acceptance_pending=True)
        (out/'fit-terminal.json').write_text(json.dumps(result,indent=2)+'\n');return result
    finally:opal.close()
