"""Bounded native control LoRA training, with an independent fresh-process restore.

Only a full approved dataset can enter this module. The two-update proof branch
is compared exactly, then its independently restored state continues as the ONE
candidate. Rejected Moshi pilots are never loaded or repeated.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from .lora import install,adapter_state,scene_loss,selected_forward
from .native_cache import load_native
from .state import load,save

RECIPE=dict(rank=16,alpha=16,projections=['q_proj','v_proj'],epochs=8,microbatch=1,
    accumulation=4,updates=304,learning_rate=.0001,warmup_updates=16,
    adam_betas=[.9,.999],adam_eps=1e-8,weight_decay=.01,gradient_clip=1.,
    ordinary_kl_weight=.1,ordinary_hidden_weight=.01,seed=20261026,
    audio_encoder_frozen=True,tts_frozen=True,base_llm_frozen=True,
    acoustic_loss_trained=False)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order(rows):
    train=sorted([r for r in rows if r['fold']=='TRAIN'],key=lambda r:r['id'])
    assert len(train)==152
    batches=[]
    for epoch in range(RECIPE['epochs']):
        shuffled=list(train);random.Random(RECIPE['seed']+epoch).shuffle(shuffled)
        batches.extend(shuffled[i:i+4] for i in range(0,len(shuffled),4))
    assert len(batches)==304 and all(len(b)==4 for b in batches)
    return batches


def learning_rate(step):
    if step<RECIPE['warmup_updates']:return RECIPE['learning_rate']*(step+1)/RECIPE['warmup_updates']
    return RECIPE['learning_rate']*.5*(1+math.cos(math.pi*(step-RECIPE['warmup_updates'])/(RECIPE['updates']-RECIPE['warmup_updates'])))


def read_example(cache,row):
    path=cache/row['file'];assert digest(path)==row['sha256']
    return torch.load(path,map_location='cpu',weights_only=False)


def train_step(llm,opt,cache,batch,step):
    opt.zero_grad(set_to_none=True)
    for group in opt.param_groups:group['lr']=learning_rate(step)
    metrics=[];before=time.monotonic()
    for row in batch:
        ex=read_example(cache,row);loss,parts=scene_loss(llm,ex,'cuda')
        if not torch.isfinite(loss):raise ValueError('Nonfinite native loss')
        (loss/len(batch)).backward();metrics.append(parts)
    params=[p for p in llm.parameters() if p.requires_grad]
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):raise ValueError('Missing/nonfinite adapter gradient')
    norm=torch.nn.utils.clip_grad_norm_(params,RECIPE['gradient_clip'])
    if not torch.isfinite(norm) or norm<=0:raise ValueError('No meaningful finite native gradient')
    opt.step();torch.cuda.synchronize()
    return dict(step=step+1,seconds=time.monotonic()-before,gradient_norm=float(norm),lr=learning_rate(step),
        ce=sum(x['ce'] for x in metrics)/len(metrics),kl=sum(x['kl'] for x in metrics)/len(metrics),hidden=sum(x['hidden'] for x in metrics)/len(metrics),
        ids=[r['id'] for r in batch],families=[r['family'] for r in batch])


def dev_loss(llm,cache,rows):
    groups=defaultdict(list)
    with torch.no_grad():
        for row in rows:
            if row['fold']=='DEV':
                _,parts=scene_loss(llm,read_example(cache,row),'cuda');groups[row['family']].append(parts['ce'])
    assert sum(map(len,groups.values()))==36
    return {k:sum(v)/len(v) for k,v in groups.items()}


def assert_deadline(path,seconds_needed=0):
    lease=json.loads(Path(path).read_text())
    if time.time()+seconds_needed>=lease['stop_unix']-180:raise TimeoutError('Preserve checkpoint/copyback reserve before native STOP')


def main(a):
    cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());assert not manifest['proof_only']
    approval=json.loads(Path(a.approval).read_text());assert approval['approved'] and approval['manifest_sha256']==manifest['source_manifest_sha256']
    assert len(manifest['scenes'])==188 and sum(r['fold']=='TRAIN' for r in manifest['scenes'])==152
    out=Path(a.output);out.mkdir(parents=True,exist_ok=(a.stage=='restore-continue'))
    metadata=dict(recipe=RECIPE,cache_sha256=digest(cache/'manifest.json'),approval_sha256=digest(a.approval),source=a.source,model_revision=manifest['model_revision'])
    if a.stage=='start':(out/'recipe.json').write_text(json.dumps(metadata,indent=2)+'\n')
    else:assert json.loads((out/'recipe.json').read_text())==metadata
    assert_deadline(a.lease,300)
    random.seed(RECIPE['seed']);np.random.seed(RECIPE['seed']);torch.manual_seed(RECIPE['seed']);torch.cuda.manual_seed_all(RECIPE['seed'])
    model,duplex=load_native(a.model);llm=model.llm
    torch.manual_seed(RECIPE['seed']);torch.cuda.manual_seed_all(RECIPE['seed'])
    assert len(llm.model.layers)==36 and llm.config.attention_dropout==0
    checkrow=next(r for r in manifest['scenes'] if r['fold']=='TRAIN' and r['family']=='ordinary')
    check=read_example(cache,checkrow)
    with torch.no_grad():before=selected_forward(llm,check,'cuda')[0].cpu()
    spec=install(llm,RECIPE['rank'],RECIPE['alpha'])
    assert spec['parameters']==7667712 and len(spec['modules'])==72
    with torch.no_grad():after=selected_forward(llm,check,'cuda')[0].cpu()
    if not torch.equal(before,after):raise ValueError('Zero LoRA must match frozen native base exactly')
    future=dict(check);future['embeddings']=check['embeddings'].clone();future['embeddings'][-1]+=1
    with torch.no_grad():future_logits=selected_forward(llm,future,'cuda')[0].cpu()
    if not torch.equal(after,future_logits):raise ValueError('Future embedding changed earlier labelled predictions')
    frozen_versions={n:p._version for n,p in llm.named_parameters() if not p.requires_grad}
    llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});llm.train()
    opt=torch.optim.AdamW([p for p in llm.parameters() if p.requires_grad],lr=RECIPE['learning_rate'],betas=tuple(RECIPE['adam_betas']),eps=RECIPE['adam_eps'],weight_decay=RECIPE['weight_decay'])
    batches=order(manifest['scenes']);start=0
    def frozen_ok():
        if any(p._version!=frozen_versions[n] for n,p in llm.named_parameters() if n in frozen_versions):raise ValueError('Frozen native weight mutated')
    if a.stage=='start':
        (out/'numerical-zero-proof.json').write_text(json.dumps(dict(zero_adapter_max_error=0,causal_future_max_error=0,adapter=spec),indent=2)+'\n')
        baseline=dev_loss(llm,cache,manifest['scenes']);(out/'dev-baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
        first=train_step(llm,opt,cache,batches[0],0);frozen_ok();save(out/'step-0001.pt',llm,opt,1,metadata)
        rng_probe=dict(python=random.random(),numpy=float(np.random.rand()),cuda=torch.rand(8,device='cuda').cpu().tolist())
        second=train_step(llm,opt,cache,batches[1],1);frozen_ok();save(out/'expected-step-0002-proof-only.pt',llm,opt,2,metadata)
        (out/'expected-step2.json').write_text(json.dumps(dict(first=first,second=second,rng_probe=rng_probe),indent=2)+'\n')
        # Process exits here. Next command loads model+Adam+RNG from step1 in a NEW process.
        print(json.dumps(dict(stage='start',actual_updates=2,candidate_saved_at=1,next='fresh-process restore-continue')),flush=True);return
    start=load(out/'step-0001.pt',llm,opt,metadata);assert start==1
    rng_probe=dict(python=random.random(),numpy=float(np.random.rand()),cuda=torch.rand(8,device='cuda').cpu().tolist())
    expected=json.loads((out/'expected-step2.json').read_text());assert rng_probe==expected['rng_probe']
    second=train_step(llm,opt,cache,batches[1],1);frozen_ok()
    reference=torch.load(out/'expected-step-0002-proof-only.pt',map_location='cpu',weights_only=False)
    actual=adapter_state(llm)
    assert all(torch.equal(v,actual[n]) for n,v in reference['adapter'].items()),'Fresh-process continuation differs'
    assert all(second[k]==expected['second'][k] for k in ('ce','kl','hidden','gradient_norm','lr'))
    # Adam tensors must match as well as the adapter weights.
    state=opt.state_dict()
    for ident,old in reference['optimizer']['state'].items():
        for key,value in old.items():
            now=state['state'][ident][key]
            assert torch.equal(value.cpu(),now.cpu()) if torch.is_tensor(value) else value==now
    save(out/'step-0002.pt',llm,opt,2,metadata)
    (out/'fresh-process-restore.json').write_text(json.dumps(dict(passed=True,adapter_error=0,adam_error=0,rng_equal=True,loss_equal=True,actual_new_process=True,independent_continuation=True),indent=2)+'\n')
    log=(out/'training.jsonl').open('x');log.write(json.dumps(expected['first'])+'\n'+json.dumps(second)+'\n');log.flush()
    seconds=[expected['first']['seconds'],second['seconds']]
    try:
        # A measured admission check, not a promise that304 steps fit any GPU window.
        assert_deadline(a.lease,max(seconds)*1.3*(304-2)+600)
        for step in range(2,304):
            assert_deadline(a.lease,max(seconds[-8:])*1.5+120)
            row=train_step(llm,opt,cache,batches[step],step);frozen_ok();seconds.append(row['seconds']);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if (step+1)%19==0 or step==303:
                save(out/f'step-{step+1:04d}.pt',llm,opt,step+1,metadata)
            if (step+1)%76==0:
                result=dict(step=step+1,full36_dev_teacher_forced=dev_loss(llm,cache,manifest['scenes']),native_behavior_qualified=False)
                (out/f'dev-{step+1:04d}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        (out/'training-terminal.json').write_text(json.dumps(dict(actual_updates=304,completed=True,adopted=False,behavioral_evaluation_required=True),indent=2)+'\n')
    except BaseException:
        completed=locals().get('step',1)
        # Save actual current state on a budget/readiness stop, with honest completed count.
        save(out/'interrupted-state.pt',llm,opt,len(seconds),metadata)
        raise
    finally:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['start','restore-continue'],required=True);p.add_argument('--model',required=True);p.add_argument('--cache',required=True);p.add_argument('--approval',required=True);p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--lease',required=True);main(p.parse_args())
