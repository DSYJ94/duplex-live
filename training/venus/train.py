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
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from contextlib import nullcontext
from .lora import install,adapter_state,scene_loss,selected_forward
from .native_cache import load_native
from .state import load,save
from .offload import ActivationOffload

RECIPE=dict(rank=64,alpha=64,last_layers=1,projections=['q_proj','v_proj'],epochs=8,microbatch=1,
    accumulation=4,updates=304,learning_rate=.0001,warmup_updates=16,
    adam_betas=[.9,.999],adam_eps=1e-8,weight_decay=.01,gradient_clip=1.,
    ordinary_kl_weight=.1,ordinary_hidden_weight=.01,seed=20261026,
    audio_encoder_frozen=True,tts_frozen=True,base_llm_frozen=True,
    acoustic_loss_trained=False,forward_path='native-frozen-prefix-v1',gradient_checkpointing=False,
    deterministic_algorithms=True,cublas_workspace=':4096:8',activation_offload=False,language_only_cached_training=True,
    activation_offload_minimum_bytes=256*1024,activation_offload_cpu_limit_bytes=48*1024**3)
MAXIMUM_RECIPE=dict(RECIPE)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def order(rows):
    train=sorted([r for r in rows if r['fold']=='TRAIN'],key=lambda r:r['id'])
    count=RECIPE.get('train_count',152);assert count in (152,216) and len(train)==count
    batches=[]
    for epoch in range(RECIPE['epochs']):
        shuffled=list(train);random.Random(RECIPE['seed']+epoch).shuffle(shuffled)
        batches.extend(shuffled[i:i+4] for i in range(0,len(shuffled),4))
    assert len(batches)==count//4*RECIPE['epochs'] and all(len(b)==4 for b in batches)
    return batches


def learning_rate(step):
    if step<RECIPE['warmup_updates']:return RECIPE['learning_rate']*(step+1)/RECIPE['warmup_updates']
    return RECIPE['learning_rate']*.5*(1+math.cos(math.pi*(step-RECIPE['warmup_updates'])/(RECIPE['updates']-RECIPE['warmup_updates'])))


def read_example(cache,row):
    path=cache/row['file'];assert digest(path)==row['sha256']
    return torch.load(path,map_location='cpu',weights_only=False)


def train_step(llm,opt,cache,batch,step,update=True,control=None,opal=None):
    if opal and control:raise ValueError("OPAL and bounded LR control cannot own the same optimizer")
    torch.cuda.reset_peak_memory_stats()
    opt.zero_grad(set_to_none=True)
    multiplier=control.state['lr_multiplier'] if control else 1.
    clip=control.state['gradient_clip'] if control else RECIPE['gradient_clip']
    if not opal:
        for group in opt.param_groups:group['lr']=learning_rate(step)*multiplier
    rate_before=float(opt.param_groups[0]['lr'])
    metrics=[];losses=[];before=time.monotonic()
    for row in batch:
        ex=read_example(cache,row)
        offload=ActivationOffload(llm,RECIPE['activation_offload_minimum_bytes'],RECIPE['activation_offload_cpu_limit_bytes'])
        with offload.context() if RECIPE['activation_offload'] else nullcontext():
            loss,parts=scene_loss(llm,ex,'cuda')
            if not torch.isfinite(loss):raise ValueError('Nonfinite native loss')
            (loss/len(batch)).backward()
        losses.append(loss.detach())
        parts.update(offloaded_bytes=offload.saved_bytes,offloaded_tensors=offload.tensors);metrics.append(parts)
    params=[p for p in llm.parameters() if p.requires_grad]
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):raise ValueError('Missing/nonfinite adapter gradient')
    if opal and update:
        from .native_optimizer import gradient_norm
        norm=gradient_norm(params)
    else:norm=torch.nn.utils.clip_grad_norm_(params,clip)
    if not torch.isfinite(norm) or norm<=0:raise ValueError('No meaningful finite native gradient')
    from .diagnostics import aggregate
    family_metrics=aggregate([(row['family'],parts,1) for row,parts in zip(batch,metrics)])
    if opal and update:opal.training_batch(family_metrics)
    program=None
    if update:
        if opal:program=opal.step(torch.stack(losses).mean(),step)
        else:opt.step()
    torch.cuda.synchronize()
    result=dict(step=step+1,optimizer_updated=update,seconds=time.monotonic()-before,peak_gpu_bytes=torch.cuda.max_memory_allocated(),gradient_norm=float(norm),lr=rate_before,
        lr_multiplier=multiplier,gradient_clip=clip,
        ce=sum(x['ce'] for x in metrics)/len(metrics),kl=sum(x['kl'] for x in metrics)/len(metrics),hidden=sum(x['hidden'] for x in metrics)/len(metrics),offloaded_bytes=sum(x['offloaded_bytes'] for x in metrics),
        ids=[r['id'] for r in batch],families=[r['family'] for r in batch],family_train_metrics=family_metrics)
    if opal:
        if program and program['program_version']>0:result.update(gradient_clip=None,lr_multiplier=None,starting_recipe_gradient_clip=clip)
        result.update(opal=program,optimizer_owner='upstream-opal',lr_after_scheduler=float(opt.param_groups[0]['lr']),lr_report_scope='before OPAL step; updater program is authoritative if it changes LR',objective=float(torch.stack(losses).mean()))
    return result


def dev_loss(llm,cache,rows,with_details=False):
    from .diagnostics import aggregate
    measured=[]
    with torch.no_grad():
        for row in rows:
            if row['fold']=='DEV':
                _,parts=scene_loss(llm,read_example(cache,row),'cuda');measured.append((row['family'],parts,1))
    assert len(measured)==RECIPE.get('dev_count',36)
    details=aggregate(measured);values={k:v['ce'] for k,v in details.items()}
    return dict(ce=values,per_family=details) if with_details else values


def assert_deadline(path,seconds_needed=0):
    if os.environ.get('DUPLEX_LIVE_GUARD_FILE'):
        from portable.venus.native_app import ensure_guard
        ensure_guard()
    lease=json.loads(Path(path).read_text())
    if time.time()+seconds_needed>=lease['stop_unix']-180:raise TimeoutError('Preserve checkpoint/copyback reserve before native STOP')


def admit_epochs(step_seconds,dev_seconds,available_seconds):
    """Throughput only: no loss/holdout selection, no omitted TRAIN scenes."""
    count=RECIPE.get('train_count',152);steps=count//4
    maximum=RECIPE.get('maximum_complete_epochs',8)
    if type(maximum) is not int or not 1<=maximum<=8:raise ValueError('Explicit complete-epoch limit must be between one and eight')
    for epochs in range(maximum,0,-1):
        required=steps*epochs*step_seconds*1.35+(epochs if RECIPE.get('opal') else (epochs+1)//2)*dev_seconds*1.25+RECIPE.get('post_training_reserve_seconds',600)
        if required<available_seconds:
            return dict(epochs=epochs,updates=steps*epochs,benchmark_step_seconds=step_seconds,
                        baseline_dev_seconds=dev_seconds,available_seconds=available_seconds,
                        conservative_required_seconds=required,minimum_complete_epochs=1,maximum_complete_epochs=maximum,
                        train_scenes_per_epoch=count,all152_train_scenes_every_epoch=count==152,all_train_scenes_every_epoch=True,selection_uses_loss_or_heldout=False)
    raise TimeoutError('No complete TRAIN epoch fits; no optimizer work admitted')


def main(a):
    parent_checkpoint=getattr(a,'parent_checkpoint',None);parent_state=None;parent_steps=0
    reactive=bool(getattr(a,'reactive',False))
    opal_source=getattr(a,'opal_source',None)
    if opal_source and reactive:raise ValueError('Choose OPAL or bounded control, never both')
    if opal_source:
        from .opal_adapter import load_upstream,Options
        Options(total_steps=1,goal='Native training',backend=getattr(a,'opal_backend','keep'),maximum_agent_cost_usd=getattr(a,'opal_agent_budget',0.)).validate()
        load_upstream(opal_source)
        lease=json.loads(Path(a.lease).read_text())
        from .paced_guard import execution_plan
        execution=json.loads(execution_plan(lease).read_text())
        if execution.get('new_optimizer_updates_authorized_by_this_execution')==0:
            raise ValueError('This execution is live/evaluation-only; native OPAL cannot start training')
    if reactive and not parent_checkpoint:raise ValueError('Reactive control is scoped to the approved step190 continuation')
    if parent_checkpoint:
        from .native_optimizer import verified_parent
        assert digest(parent_checkpoint)==a.parent_sha256
        parent_state=torch.load(parent_checkpoint,map_location='cpu',weights_only=False)
        parent_steps=verified_parent(parent_state,a.parent_sha256,allow244=bool(opal_source))
        if reactive and parent_steps!=190:raise ValueError('The older bounded control is scoped to190 only')
        RECIPE.update(train_count=216,dev_count=52,epochs=8,updates=432,learning_rate=.00005,seed=20261027,post_training_reserve_seconds=3600)
        MAXIMUM_RECIPE.update(RECIPE)
    if opal_source:
        from .opal_adapter import UPSTREAM_COMMIT
        maximum_epochs=getattr(a,'maximum_epochs',8)
        if type(maximum_epochs) is not int or not 1<=maximum_epochs<=8:raise ValueError('Explicit complete-epoch limit must be between one and eight')
        RECIPE['maximum_complete_epochs']=maximum_epochs
        RECIPE['opal']=dict(upstream_commit=UPSTREAM_COMMIT,backend=getattr(a,'opal_backend','keep'),agent_budget_usd=getattr(a,'opal_agent_budget',0.),nonblocking=True,goal='Improve native duplex control and ordinary retention; full native speech and behavior decide acceptance')
        MAXIMUM_RECIPE.update(RECIPE)
    if reactive:
        RECIPE['reactive_control']='bounded-agent-lr-clip-v1'
        MAXIMUM_RECIPE.update(RECIPE)
    cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text());assert not manifest['proof_only']
    approval=json.loads(Path(a.approval).read_text());assert approval['approved'] and approval['manifest_sha256']==manifest['source_manifest_sha256']
    train_count=216 if parent_checkpoint else 152;dev_count=52 if parent_checkpoint else 36;per_epoch=train_count//4
    assert len(manifest['scenes'])==train_count+dev_count and sum(r['fold']=='TRAIN' for r in manifest['scenes'])==train_count and sum(r['fold']=='DEV' for r in manifest['scenes'])==dev_count
    assert approval['train']==train_count and approval['dev']==dev_count
    if parent_checkpoint:assert manifest['native_speech_timing_schema']=='word-aligned-native-v1'
    data_coverage=None
    if opal_source:
        if not getattr(a,'coverage_manifest',None):
            raise ValueError('Native OPAL requires the exact approved coverage manifest before model loading')
        assert digest(a.coverage_manifest)==manifest['source_manifest_sha256']
        from .diagnostics import coverage
        data_coverage=coverage(json.loads(Path(a.coverage_manifest).read_text()))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=(a.stage=='restore-continue'))
    identity=dict(cache_sha256=digest(cache/'manifest.json'),approval_sha256=digest(a.approval),source=a.source,cache_producer_source=a.cache_producer,model_revision=manifest['model_revision'])
    if parent_checkpoint:identity.update(parent_checkpoint_sha256=a.parent_sha256,parent_training_steps=parent_steps,parent_training_source=parent_state['metadata']['source'])
    metadata=None
    if a.stage=='restore-continue':
        metadata=json.loads((out/'recipe.json').read_text())
        assert all(metadata[k]==v for k,v in identity.items())
        chosen=metadata['recipe'];assert 1<=chosen['epochs']<=8 and chosen['updates']==per_epoch*chosen['epochs'] and (not parent_checkpoint or chosen['updates']<=512)
        assert isinstance(chosen['activation_offload'],bool)
        assert chosen==dict(MAXIMUM_RECIPE,epochs=chosen['epochs'],updates=chosen['updates'],activation_offload=chosen['activation_offload'])
        RECIPE.update(chosen)
    assert_deadline(a.lease,300)
    random.seed(RECIPE['seed']);np.random.seed(RECIPE['seed']);torch.manual_seed(RECIPE['seed']);torch.cuda.manual_seed_all(RECIPE['seed'])
    model,duplex=load_native(a.model,language_only=True);llm=model.llm
    random.seed(RECIPE['seed']);np.random.seed(RECIPE['seed'])
    torch.manual_seed(RECIPE['seed']);torch.cuda.manual_seed_all(RECIPE['seed'])
    assert len(llm.model.layers)==36 and llm.config.attention_dropout==0
    checkrow=next(r for r in manifest['scenes'] if r['fold']=='TRAIN' and r['family']=='ordinary')
    check=read_example(cache,checkrow)
    with torch.no_grad():before=selected_forward(llm,check,'cuda')[0].cpu()
    spec=install(llm,RECIPE['rank'],RECIPE['alpha'],RECIPE['last_layers'])
    assert spec['parameters']==851968 and len(spec['modules'])==2
    with torch.no_grad():after=selected_forward(llm,check,'cuda')[0].cpu()
    if not torch.equal(before,after):raise ValueError('Zero LoRA must match frozen native base exactly')
    import copy
    future=copy.deepcopy(check)
    if check.get('forward_path')=='native-frozen-prefix-v1':future['prefix_inputs'][-1]['hidden']+=1
    else:future['embeddings'][-1]+=1
    with torch.no_grad():future_logits=selected_forward(llm,future,'cuda')[0].cpu()
    if not torch.equal(after,future_logits):raise ValueError('Future embedding changed earlier labelled predictions')
    frozen_versions={n:p._version for n,p in llm.named_parameters() if not p.requires_grad}
    llm.gradient_checkpointing_disable();llm.eval()  # Autograd stays enabled; match native attention and KV path.
    opt=torch.optim.AdamW([p for p in llm.parameters() if p.requires_grad],lr=RECIPE['learning_rate'],betas=tuple(RECIPE['adam_betas']),eps=RECIPE['adam_eps'],weight_decay=RECIPE['weight_decay'])
    if parent_checkpoint and a.stage=='start':
        from .state import restore_parent
        proof=restore_parent(parent_state,llm,opt)
        (out/'parent-restore.json').write_text(json.dumps(dict(parent_step=parent_steps,parent_checkpoint_sha256=a.parent_sha256,**proof),indent=2)+'\n')
    batches=order(manifest['scenes']);start=0
    control=None
    if reactive:
        from .reactive import ReactiveControl
        restored=None
        if a.stage=='restore-continue':
            saved=torch.load(out/'step-0001.pt',map_location='cpu',weights_only=False)
            restored=saved['reactive_control']
        control=ReactiveControl(out/'reactive',identity,restored)
    opal=None
    def attach_opal():
        nonlocal opt,opal
        from .native_optimizer import NativeOpal
        opal=NativeOpal(opt,llm,source=opal_source,run_dir=out/('opal-start' if a.stage=='start' else 'opal-restored'),recipe=RECIPE,backend=RECIPE['opal']['backend'],agent_budget=RECIPE['opal']['agent_budget_usd'],goal=RECIPE['opal']['goal'])
        opt=opal.optimizer
    def checkpoint(path,step):
        if opal:
            from .opal_adapter import checkpoint_manifest
            assert checkpoint_manifest(opt)['update']==step
        return save(path,llm,opt,step,metadata,control.snapshot() if control else None,opal.diagnostics_state if opal else None)
    def frozen_ok():
        if any(p._version!=frozen_versions[n] for n,p in llm.named_parameters() if n in frozen_versions):raise ValueError('Frozen native weight mutated')
    if a.stage=='start':
        (out/'numerical-zero-proof.json').write_text(json.dumps(dict(zero_adapter_max_error=0,causal_future_max_error=0,adapter=spec),indent=2)+'\n')
        from .native_cache import training_input_proof
        proofrow=next(r for r in manifest['scenes'] if read_example(cache,r)['logit_proofs'])
        from .lora import disabled
        with disabled(llm):parity=training_input_proof(model,read_example(cache,proofrow))
        (out/'language-only-native-input-proof.json').write_text(json.dumps(parity,indent=2)+'\n')
        assert parity['passed'],'Language-only cached training must retain native parity'
        if a.baseline_reference:
            reference=json.loads(Path(a.baseline_reference).read_text());assert reference['cache_sha256']==identity['cache_sha256']
            baseline=reference['full36_dev_baseline'];dev_seconds=reference['conservative_baseline_dev_seconds']
            (out/'baseline-reference.json').write_text(json.dumps(reference,indent=2)+'\n')
        else:
            before_dev=time.monotonic();baseline_report=dev_loss(llm,cache,manifest['scenes'],with_details=bool(opal_source));dev_seconds=time.monotonic()-before_dev
            baseline=baseline_report['ce'] if opal_source else baseline_report
        (out/'dev-baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
        workload=[]
        for row in manifest['scenes']:
            if row['fold']=='TRAIN':
                ex=read_example(cache,row)
                score=len(ex['segments'])*(1.5 if row['family']=='ordinary' else 1)+len(ex['embeddings'])/64
                workload.append((score,row))
        calibration=[row for _,row in sorted(workload,key=lambda x:x[0],reverse=True)[:4]]
        fallback=False
        try:benchmark=train_step(llm,opt,cache,calibration,0,update=False)
        except torch.cuda.OutOfMemoryError:
            fallback=True
        if fallback:
            opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();RECIPE['activation_offload']=True
            (out/'calibration-memory-fallback.json').write_text(json.dumps(dict(no_optimizer_update=True,language_only_no_offload_oom=True,lossless_saved_activation_offload=True),indent=2)+'\n')
            benchmark=train_step(llm,opt,cache,calibration,0,update=False)
        frozen_ok();opt.zero_grad(set_to_none=True)
        benchmark['selection']='four largest TRAIN feed workloads; no loss or answer selection'
        (out/'no-update-gradient-benchmark.json').write_text(json.dumps(benchmark,indent=2)+'\n')
        admission=admit_epochs(benchmark['seconds'],dev_seconds,json.loads(Path(a.lease).read_text())['stop_unix']-time.time()-180)
        RECIPE.update(epochs=admission['epochs'],updates=admission['updates']);batches=order(manifest['scenes'])
        metadata=dict(**identity,recipe=dict(RECIPE),maximum_recipe=MAXIMUM_RECIPE,admission=admission)
        (out/'recipe.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(dict(event='complete_epoch_schedule_admitted',**admission)),flush=True)
        if opal_source:
            attach_opal();opal.development(baseline)
            if not a.baseline_reference:opal.diagnose([],0,dev=baseline_report['per_family'],data_coverage=data_coverage)
        first=train_step(llm,opt,cache,batches[0],0,control=control,opal=opal);frozen_ok();checkpoint(out/'step-0001.pt',1)
        rng_probe=dict(python=random.random(),numpy=float(np.random.rand()),cuda=torch.rand(8,device='cuda').cpu().tolist())
        second=train_step(llm,opt,cache,batches[1],1,control=control,opal=opal);frozen_ok();checkpoint(out/'expected-step-0002-proof-only.pt',2)
        (out/'expected-step2.json').write_text(json.dumps(dict(first=first,second=second,rng_probe=rng_probe),indent=2)+'\n')
        # Process exits here. Next command loads model+Adam+RNG from step1 in a NEW process.
        print(json.dumps(dict(stage='start',actual_updates=2,candidate_saved_at=1,next='fresh-process restore-continue')),flush=True)
        if opal:opal.close()
        return
    if opal_source:attach_opal()
    start=load(out/'step-0001.pt',llm,opt,metadata,opal.restore_diagnostics if opal else None);assert start==1
    rng_probe=dict(python=random.random(),numpy=float(np.random.rand()),cuda=torch.rand(8,device='cuda').cpu().tolist())
    expected=json.loads((out/'expected-step2.json').read_text());assert rng_probe==expected['rng_probe']
    second=train_step(llm,opt,cache,batches[1],1,control=control,opal=opal);frozen_ok()
    reference=torch.load(out/'expected-step-0002-proof-only.pt',map_location='cpu',weights_only=False)
    actual=adapter_state(llm)
    assert all(torch.equal(v,actual[n]) for n,v in reference['adapter'].items()),'Fresh-process continuation differs'
    assert all(second[k]==expected['second'][k] for k in ('ce','kl','hidden','gradient_norm','lr'))
    # Adam tensors must match as well as the adapter weights.
    state=opt.state_dict()
    if control:assert reference['reactive_control']==control.snapshot(),'Fresh-process reactive state differs'
    from .native_optimizer import assert_optimizer_equal
    assert_optimizer_equal(reference['optimizer'],state)
    if opal:assert reference['training_diagnostics']==opal.diagnostics_state
    checkpoint(out/'step-0002.pt',2)
    (out/'fresh-process-restore.json').write_text(json.dumps(dict(passed=True,adapter_error=0,adam_error=0,rng_equal=True,loss_equal=True,actual_new_process=True,independent_continuation=True),indent=2)+'\n')
    log=(out/'training.jsonl').open('x');log.write(json.dumps(expected['first'])+'\n'+json.dumps(second)+'\n');log.flush()
    seconds=[expected['first']['seconds'],second['seconds']]
    recent=[expected['first'],second]
    if control:control.observe(2,recent)
    try:
        for step in range(2,RECIPE['updates']):
            assert_deadline(a.lease,max(seconds[-8:])*1.5+120)
            if control:control.boundary(step)
            row=train_step(llm,opt,cache,batches[step],step,control=control,opal=opal);frozen_ok();seconds.append(row['seconds']);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            recent.append(row);recent=recent[-8:]
            if control and (step+1)%8==0:control.observe(step+1,recent)
            if opal and (step+1)%8==0:opal.diagnose(recent,step+1,data_coverage=data_coverage)
            if (step+1)%per_epoch==0 or step==RECIPE['updates']-1:
                checkpoint(out/f'step-{step+1:04d}.pt',step+1)
            if (step+1)%(per_epoch if opal else per_epoch*2)==0 or step==RECIPE['updates']-1:
                dev_report=dev_loss(llm,cache,manifest['scenes'],with_details=bool(opal))
                result=dict(step=step+1,parent_training_steps=parent_steps,dev_scenes=dev_count,full_dev_teacher_forced=dev_report['ce'] if opal else dev_report,native_behavior_qualified=False)
                if opal:result['full_dev_group_metrics']=dev_report['per_family']
                if not parent_checkpoint:result['full36_dev_teacher_forced']=result['full_dev_teacher_forced']
                (out/f'dev-{step+1:04d}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
                if opal:
                    opal.development(result['full_dev_teacher_forced'])
                    opal.diagnose(recent,step+1,dev=dev_report['per_family'],data_coverage=data_coverage)
                    checkpoint(out/f'step-{step+1:04d}.pt',step+1)
                if control:
                    control.observe(step+1,recent,result['full_dev_teacher_forced'])
                    checkpoint(out/f'step-{step+1:04d}.pt',step+1)
        (out/'training-terminal.json').write_text(json.dumps(dict(actual_updates=RECIPE['updates'],parent_training_steps=parent_steps,total_optimizer_updates=RECIPE['updates']+parent_steps,actual_complete_epochs=RECIPE['epochs'],maximum_authorized_updates=512 if parent_checkpoint else 304,completed=True,adopted=False,behavioral_evaluation_required=True),indent=2)+'\n')
    except BaseException:
        completed=locals().get('step',1)
        # Save actual current state on a budget/readiness stop, with honest completed count.
        checkpoint(out/'interrupted-state.pt',len(seconds))
        raise
    finally:
        log.close()
        if opal:opal.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['start','restore-continue'],required=True);p.add_argument('--model',required=True);p.add_argument('--cache',required=True);p.add_argument('--approval',required=True);p.add_argument('--output',required=True);p.add_argument('--source',required=True);p.add_argument('--cache-producer',required=True);p.add_argument('--lease',required=True);p.add_argument('--baseline-reference');p.add_argument('--parent-checkpoint');p.add_argument('--parent-sha256');p.add_argument('--reactive',action='store_true',help='Nonblocking bounded agent decisions; no external API or executable patches');p.add_argument('--opal-source',help='Exact clean pinned OPAL checkout; optional native optimizer owner');p.add_argument('--opal-backend',default='keep',help='keep by default; explicit bounded mailbox only');p.add_argument('--opal-agent-budget',type=float,default=0.);p.add_argument('--maximum-epochs',type=int,choices=range(1,9),default=8,help='Complete-epoch admission cap for OPAL; never truncate a corpus');p.add_argument('--coverage-manifest',help='Exact approved source manifest for TRAIN/DEV aggregate coverage only');main(p.parse_args())
