"""Fresh matched PersonaPlex/Context Spanning lineages, with real native state.

No saved Venus/native520 adapter is used as an initialization. Original520
word/clock targets remain protected objectives. Native behavior and speech,
including the real backend, still determine adoption after these training runs.
"""
import argparse
import json,os,subprocess,sys,time
from pathlib import Path

from evals.moshi_model_download import sha256 as sha
from training.personaplex import matched_guard as guard
from training.personaplex.matched_identity import lineage_source,reference_source_matches,validate_lineage
from training.personaplex.matched_schedule import SIZES,STEPS,EPOCHS,validate,epoch

BASE=Path('/projects/u6ada/duplex-live')
BASES=('personaplex','context-spanning')
CAPACITIES={'narrow':1894400,'broad':105070592}
RECIPE=dict(updates=STEPS*EPOCHS,epochs=EPOCHS,steps_per_epoch=STEPS,
    learning_rate=2e-6,warmup_updates=16,gradient_clip=1.,adam_betas=[.9,.999],
    adam_eps=1e-8,weight_decay=.01,foreach=False,fused=False,amsgrad=False,
    source_precision='BF16 frozen base; FP32 adapters and Adam',attention='MATH',
    actual_response_first_word_weight=16.,reference_KL=1.,reference_hidden_MSE=10.,
    word_anchor=64.,clock_anchor=64.,RET_class_balanced_coefficient=8.,
    stream_autograd='recorded call boundaries; unchanged native KV values',
    full_unstreamed_objectives_preserved=True,optimizer_owner='upstream OPAL',
    adaptive_policy='common keep for matched initial comparison; no model API',
    sampling='all323 rows each complete49-step epoch; exact per-component inverse visits',
    seed=42424242,external_model_API_calls=0)


def write(folder,name,value):
    with (Path(folder)/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def corpus(folder,expected):
    import torch
    from evals.context_native_audio49_data import digest
    report=json.loads((folder/'inputs.json').read_text())
    if (sha(folder/'inputs.json')!=expected or not report['complete323'] or report['groups']!=SIZES
            or report['training_approved'] is not False or report['optimizer_updates']!=0
            or sha(folder/'inputs.pt')!=report['input_sha256']):
        raise ValueError('Exact full323 CPU-qualified input artifact required')
    builder=subprocess.check_output(['git','show',report['source']+':training/personaplex/matched_inputs.py'])
    if builder!=Path(__file__).with_name('matched_inputs.py').read_bytes():
        raise ValueError('Original data builder changed; no relabelled input approval')
    groups=torch.load(folder/'inputs.pt',weights_only=True,map_location='cpu');validate(groups)
    actual={g:[dict(id=r['id'],tensors={k:digest(v) for k,v in r.items() if isinstance(v,torch.Tensor)})
        for r in rows] for g,rows in groups.items()}
    if actual!=report['input_tensor_fingerprints'] or [epoch(groups,e) for e in range(EPOCHS)]!=report['complete_four_epoch_schedules']:
        raise ValueError('Full tensor ownership or complete schedule changed')
    return groups,report


def worst_rows(groups):
    def length(row):return max(row.get('inputs',()).shape[-1] if 'inputs' in row else 0,
        row.get('codes',()).shape[-1] if 'codes' in row else 0)
    return {g:max(range(len(rows)),key=lambda i:(length(rows[i]),rows[i]['id'])) for g,rows in groups.items()}


def metadata(a,report):
    return dict(schema='dl-moshi-matched-lineage-v1',source=lineage_source(a),base=a.base,capacity=a.capacity,
        data_report_sha256=a.inputs_sha,input_sha256=report['input_sha256'],recipe=RECIPE,
        trainable_parameters=CAPACITIES[a.capacity],old520_adapter_used_as_initialization=False,
        old520_word_clock_targets_preserved=True,own_public_base_four_reference_retention=True,adopted=False)


def fingerprint(model,groups):
    import torch
    from evals.context_native_audio49_data import digest
    result={}
    with torch.no_grad():
        for row in groups['reference']:
            mask=row['mask'].cuda();h,z=model.forward_codes(row['inputs'].cuda())
            result[row['id']]=dict(hidden=digest(h[:,mask]),text_logits=digest(z[0,0,mask]))
    return result


def model(a,folder):
    import torch
    from evals.context_spanning_download import SOURCE_COMMIT,WEIGHT_SHA256
    from evals.personaplex_train import MODEL_HASHES
    from evals.context_spanning_probe import audited_loads
    from evals.context_execution import math_attention
    from training.personaplex.foundation_context import raw_model
    pp=BASE/'public-models/personaplex-fdaf-v1';vendor=BASE/'vendor/context-spanning-79df-v1'
    weights_dir=BASE/'public-models/context-spanning-e62a-v1'
    if (subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(vendor),'status','--porcelain'],text=True).strip()
            or any(sha(pp/name)!=value for name,value in MODEL_HASHES.items())
            or (a.base=='context-spanning' and sha(weights_dir/'context_spanning_7b.pt')!=WEIGHT_SHA256)):
        raise ValueError('Exact public base and unmodified native vendor required')
    os.environ.update(CS_BASE_DIR=str(pp),CS_WEIGHTS_DIR=str(weights_dir),CS_VOICES_DIR=str(weights_dir/'voices'),HF_HUB_OFFLINE='1')
    sys.path.insert(0,str(vendor))
    os.environ['TRITON_LIBCUDA_PATH']=subprocess.check_output([sys.executable,'-B','-m','evals.cuda_linker_path','--out',str(folder/'driver')],text=True).strip()
    from contextspan.model import weights
    from contextspan.model.engine import seed_all
    math_attention();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    seed_all(RECIPE['seed']);loads=[]
    with audited_loads(loads):
        if a.base=='personaplex':lm=raw_model(weights,'cuda')
        else:
            lm,mimi,_=weights.load_model(str(weights_dir/'context_spanning_7b.pt'),'cuda')
            del mimi;torch.cuda.empty_cache()
    last=[r for r in loads if r['module_identity']==id(lm)]
    if (not last or not last[-1]['strict'] or last[-1]['missing'] or last[-1]['unexpected']
            or lm.dep_q!=16 or len(lm.transformer.layers)!=32):
        raise ValueError('Strict native model/all16 depth heads/32 temporal layers required')
    return lm.eval().requires_grad_(False),last


def configure(lm,capacity):
    import torch
    from evals.personaplex_adapters import install_adapters,adapter_parameters
    from training.personaplex.capacity import configure as broad
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        if capacity=='narrow':install_adapters(lm,last_layers=4,rank=8,scaling=2.)
        else:broad(lm,'broad-lora64')
    selected=adapter_parameters(lm)
    if (sum(p.numel() for p in selected.values())!=CAPACITIES[capacity]
            or set(selected)!={n for n,p in lm.named_parameters() if p.requires_grad}
            or any(p.dtype!=torch.float32 for p in selected.values())):
        raise ValueError('Exact declared new-capacity coefficient ownership required')
    return selected


def runtime(lm,selected,a,folder,programs=None):
    import torch
    from training.venus.native_optimizer import NativeOpal
    base=torch.optim.AdamW(selected.values(),lr=RECIPE['learning_rate'],betas=tuple(RECIPE['adam_betas']),
        eps=RECIPE['adam_eps'],weight_decay=RECIPE['weight_decay'],foreach=False,fused=False,amsgrad=False)
    # Adam exists BEFORE the wrapper/scheduler, and owns these exact parameters.
    from training.personaplex.matched_state import materialize_adam
    materialized=materialize_adam(base) if programs is None else None
    active=NativeOpal(base,lm,source=BASE/'opal-upstream-v1',run_dir=folder/'opal',
        recipe=RECIPE,backend='keep',agent_budget=0.,initial_program=programs)
    active.adam_materialization=materialized
    return active


def gradients(lm,selected,*,acoustic=False):
    import torch
    from training.venus.native_optimizer import gradient_norm
    needed={n:p for n,p in selected.items() if not acoustic or not n.startswith('text_linear.')}
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in needed.values()):
        raise ValueError('Missing or nonfinite actual native adapter gradient')
    if any(p.grad is not None for n,p in lm.named_parameters() if n not in selected):
        raise ValueError('Frozen source parameter acquired a gradient')
    if acoustic and any(p.grad is not None and p.grad.count_nonzero() for n,p in selected.items() if n.startswith('text_linear.')):
        raise ValueError('Assistant audio acquired text-head supervision')
    paths=sum(bool(p.grad.count_nonzero()) for n,p in needed.items() if '.lora_B.' in n)
    if paths!=sum('.lora_B.' in n for n in needed):raise ValueError('A declared temporal/text adapter B path is dead')
    norm=gradient_norm(list(needed.values()))
    if not torch.isfinite(norm) or norm<=0:raise ValueError('Native gradient has no finite useful norm')
    return dict(gradient_norm=float(norm),nonzero_B_paths=paths,all_trainable_gradients_present_finite=True)


def frozen_versions(lm):return {n:p._version for n,p in lm.named_parameters() if not p.requires_grad}


def check_frozen(lm,before):
    if frozen_versions(lm)!=before:raise ValueError('Frozen native foundation changed')


def checkpoint(a,path,lm,active,meta,references,groups):
    from training.personaplex.matched_state import save
    from training.venus.state import rng_state,exact_state
    before=rng_state();value=save(path,lm,active,meta,references)
    values=fingerprint(lm,groups)
    if not exact_state(before,rng_state()):raise ValueError('Native fingerprint changed checkpoint RNG')
    record=dict(source=a.source,checkpoint_sha256=value,fingerprints=values,
        execution_id=os.environ['DUPLEX_NATIVE_EXECUTION_ID'],step=active.optimizer.update)
    target=Path(str(path)+'.native-reference.json');temporary=Path(str(target)+'.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n');os.replace(temporary,target)
    return value


def numerical(lm,groups):
    import torch
    from evals.context_training_parity import precise_native
    from evals.native_control_reference import linear_reference
    from evals.context_native_audio49_audit import depth_checks
    from evals.context_native_audio49_training import forward
    from evals.native_control_numerical import target_positions
    result=[]
    # These are NEW four-worst native checks, not a relabelled old49 report.
    # The full immutable input validators still cover every323 component.
    with torch.no_grad(),precise_native(lm),linear_reference():
        for kind in ('private','simple','clarify','cancel'):
            row=max((r for r in groups['audio49'] if r['kind']==kind),key=lambda r:r['codes'].shape[-1])
            codes=row['codes'].cuda();mask=row['audio_mask'][0].cuda()
            output=forward(lm,codes);positions,counts=target_positions(mask,codes,output)
            parity=depth_checks(lm,codes,positions)
            if not parity['passed']:raise ValueError('New-base native depth parity failed')
            result.append(dict(id=row['id'],kind=kind,depth_parity=parity,assistant_audio_targets=counts))
            del codes,mask,output
    return result


def admit(a,groups,report,folder):
    import torch
    from evals.context_execution import snapshot
    from evals.context_native_audio49_data import digest
    from training.personaplex.matched_losses import Objective
    guard.check(a.lease,1800);lm,loads=model(a,folder);references={}
    baseline=fingerprint(lm,groups)
    with torch.no_grad():
        for row in groups['reference']:
            mask=row['mask'].cuda();h,z=lm.forward_codes(row['inputs'].cuda())
            references[row['id']]=(h[:,mask].detach().cpu().clone(),z[0,0,mask].float().softmax(-1).detach().cpu())
    selected=configure(lm,a.capacity)
    if fingerprint(lm,groups)!=baseline:raise ValueError('Zero-B expansion changed native outputs')
    parity=numerical(lm,groups)
    frozen=frozen_versions(lm);before={n:digest(p) for n,p in selected.items()}
    objective=Objective(lm,groups,references);rows=[];torch.cuda.reset_peak_memory_stats()
    # Four predeclared longest original task kinds, with actual native acoustic
    # backward and complete temporal B ownership; text-head gradient stays absent.
    for kind in ('private','simple','clarify','cancel'):
        index=max((i for i,r in enumerate(groups['audio49']) if r['kind']==kind),key=lambda i:groups['audio49'][i]['codes'].shape[-1])
        for p in selected.values():p.grad=None
        started=time.monotonic()
        part=objective.backward(dict(group='audio49',index=index,id=groups['audio49'][index]['id'],weight=1.,visits_this_epoch=1))
        torch.cuda.synchronize();rows.append(dict(kind=kind,seconds=time.monotonic()-started,**part,**gradients(lm,selected,acoustic=True)))
    # Time a complete worst-geometry eleven-objective update, without stepping.
    for p in selected.values():p.grad=None
    chosen=worst_rows(groups);parts=[];started=time.monotonic()
    for group,index in chosen.items():
        guard.check(a.lease,300)
        exposure=next(x for step in epoch(groups,0) for x in step if x['group']==group and x['index']==index)
        parts.append(objective.backward(exposure))
    torch.cuda.synchronize();seconds=time.monotonic()-started;full_gradient=gradients(lm,selected)
    check_frozen(lm,frozen)
    if before!={n:digest(p) for n,p in selected.items()}:raise ValueError('Admission changed trainable parameters')
    for p in selected.values():p.grad=None
    active=runtime(lm,selected,a,folder)
    try:
        adam=active.adam_materialization;meta=metadata(a,report)
        init=folder/'init.pt';t=time.monotonic();state_sha=checkpoint(a,init,lm,active,meta,references,groups);save_seconds=time.monotonic()-t
        proof=dict(schema='dl-moshi-matched-admission-v1',source=a.source,base=a.base,capacity=a.capacity,
            data_report_sha256=a.inputs_sha,pid=os.getpid(),execution_id=os.environ['DUPLEX_NATIVE_EXECUTION_ID'],
            actual_CUDA=True,zero_B_native_outputs_exact=True,baseline_fingerprints=baseline,
            strict_loads=loads,execution=snapshot(lm),trainable_parameters=CAPACITIES[a.capacity],
            total_model_parameters=sum(p.numel() for p in lm.parameters()),
            frozen_base_parameters=sum(p.numel() for p in lm.parameters() if not p.requires_grad),
            original_all323_input_rows=True,new_four_worst_depth_checks=parity,
            all_four_acoustic_gradients=rows,complete_worst_update=parts,
            original_TRAIN_RET_population=objective.ret,
            worst_update_gradient=full_gradient,materialized_Adam=adam,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            worst_update_seconds=seconds,checkpoint_seconds=save_seconds,
            estimated_complete196_seconds=seconds*196*1.25+save_seconds*32+600,
            checkpoint_sha256=state_sha,optimizer_updates=0,passed=True,
            independent_fresh_restore_required=True,joint_four_arm_schedule_required=True,training_approved=False,adopted=False)
        write(folder,'admission.json',proof)
    finally:active.close()


def verified_admission(row,source,inputs_sha,read_path=Path):
    """Read original proofs without changing producer identity or executing a model."""
    proofs=[]
    for kind in ('admission','restore'):
        p=read_path(row[kind]);terminal=json.loads(read_path(row[kind+'_terminal']).read_text())
        value=json.loads(p.read_text())
        if (sha(p)!=row[kind+'_sha256'] or value['source']!=source or not value['passed']
                or not value['actual_CUDA'] or terminal['actual_exit_code']!=0
                or terminal['execution_id']!=value['execution_id'] or terminal['source']!=source):
            raise ValueError('Actual successful native stage terminal and full proof required')
        proofs.append(value)
    admission,restored=proofs
    if (admission['schema']!='dl-moshi-matched-admission-v1'
            or admission['trainable_parameters']!=CAPACITIES[row['capacity']]
            or not admission['zero_B_native_outputs_exact'] or not admission['original_all323_input_rows']
            or len(admission['all_four_acoustic_gradients'])!=4
            or {r['kind'] for r in admission['all_four_acoustic_gradients']}!={'private','simple','clarify','cancel'}
            or any(not r['all_trainable_gradients_present_finite'] for r in admission['all_four_acoustic_gradients'])
            or len(admission['new_four_worst_depth_checks'])!=4
            or any(not r['depth_parity']['passed'] for r in admission['new_four_worst_depth_checks'])
            or len(admission['complete_worst_update'])!=len(SIZES)
            or {r['group'] for r in admission['complete_worst_update']}!=set(SIZES)
            or admission['materialized_Adam']['optimizer_updates']!=0
            or admission['optimizer_updates']!=0
            or admission['base']!=row['base'] or admission['capacity']!=row['capacity']
            or admission['data_report_sha256']!=inputs_sha
            or admission['checkpoint_sha256']!=restored['checkpoint_sha256']
            or admission['execution_id']==restored['execution_id']
            or restored['reference_execution_id']!=admission['execution_id']
            or restored['restored_training_step']!=0 or not restored['all_Adam_RNG_OPAL_fields_exact']
            or not restored['native_logits_exact']):
        raise ValueError('Every fresh lineage needs its own exact independent zero-update restore')
    seconds=admission['estimated_complete196_seconds']
    if not isinstance(seconds,(int,float)) or not __import__('math').isfinite(seconds) or seconds<=0:
        raise ValueError('Actual finite measured full-schedule estimate required')
    return admission,restored,seconds


def joint_schedule(a):
    """All four actual admissions/restores before either capacity starts learning."""
    if getattr(a,'lineage_source',None):
        from training.personaplex.matched_continue_policy import evaluate
        return evaluate(a)
    path=Path(a.joint_admission)
    if sha(path)!=a.joint_admission_sha:raise ValueError('Pinned complete four-arm schedule required')
    packet=json.loads(path.read_text());entries=packet['entries']
    if (packet['source']!=a.source or packet['data_report_sha256']!=a.inputs_sha
            or len(entries)!=4 or {(r['base'],r['capacity']) for r in entries}!={(b,c) for b in BASES for c in CAPACITIES}):
        raise ValueError('All four matched lineages required, not a passing subset')
    estimates={b:0. for b in BASES}
    for row in entries:
        admission,restored,seconds=verified_admission(row,a.source,a.inputs_sha)
        complete=Path(row['training_terminal']) if row.get('training_terminal') else None
        if complete is not None and complete.exists():
            terminal=json.loads(complete.read_text())
            saved=Path(terminal['final_checkpoint'])
            reference=json.loads(Path(str(saved)+'.native-reference.json').read_text())
            if (not terminal['completed'] or terminal['source']!=a.source
                    or terminal['total_lineage_updates']!=196 or terminal['complete_epochs']!=4
                    or sha(saved)!=terminal['checkpoint_sha256']
                    or reference['checkpoint_sha256']!=terminal['checkpoint_sha256']
                    or reference['step']!=196 or reference['source']!=a.source):
                raise ValueError('Only actual complete checkpointed training can reduce remaining schedule')
            seconds=0.
        estimates[row['base']]+=seconds
    lease=guard.check(a.lease)
    available=guard.LIMIT-lease['prior_GPU_seconds']-2*(time.time()-lease['start_unix'])
    # Serial capacities on each GPU, parallel bases. Include billed idle time
    # and six aggregate GPU hours reserved for restore/evaluation/copyback.
    required=2*max(estimates.values())+6*3600
    if required>available:raise RuntimeError('Useful complete paired schedule cannot fit the carried96GPUh budget')
    return dict(estimated_serial_seconds_per_base=estimates,required_GPU_seconds=required,
        available_GPU_seconds=available,all_four_native_actual_terminals=True)


def run_training(a,lm,active,state,groups,folder):
    import signal
    import torch
    from training.personaplex.matched_losses import Objective
    from training.venus.opal_adapter import log_observation
    from training.venus.state import restore_rng
    joint=joint_schedule(a)
    independent=json.loads(Path(a.restore_proof).read_text())
    if (sha(Path(a.restore_proof))!=a.restore_proof_sha or not independent['passed']
            or independent['source']!=a.source or independent['checkpoint_sha256']!=a.checkpoint_sha
            or independent['execution_id']==os.environ['DUPLEX_NATIVE_EXECUTION_ID']
            or independent['restored_training_step']!=state['step']
            or not independent['all_Adam_RNG_OPAL_fields_exact'] or not independent['native_logits_exact']):
        raise ValueError('A distinct-process complete restore is required before any optimizer step')
    selected={n:p for n,p in lm.named_parameters() if p.requires_grad};frozen=frozen_versions(lm)
    objective=Objective(lm,groups,state['references']);meta=state['metadata']
    write(folder,'recipe.json',meta);write(folder,'joint-admission.json',joint)
    start=active.optimizer.update
    if start!=state['step'] or not 0<=start<RECIPE['updates']:raise ValueError('Actual unreplayed training state required')
    restore_rng(state['rng']);last_seconds=600.;actual=start
    old_handler=signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt('Scheduler termination; preserving actual state')))
    try:
        with (folder/'training.jsonl').open('x') as log:
            for number in range(start,RECIPE['updates']):
                lease=guard.check(a.lease)
                if time.time()+max(1200,last_seconds*1.5+900)>=lease['stop_unix']:
                    if actual==start:raise RuntimeError('Insufficient time for useful training; no zero-update resume loop')
                    value=checkpoint(a,folder/'window-state.pt',lm,active,meta,state['references'],groups)
                    write(folder,'training-window-terminal.json',dict(completed=False,source=a.source,
                        actual_new_updates=actual-start,total_lineage_updates=actual,checkpoint_sha256=value,
                        checkpoint=str(folder/'window-state.pt'),verified_postupdate_resume_required=True))
                    return
                batch=epoch(groups,number//STEPS)[number%STEPS]
                active.optimizer.zero_grad(set_to_none=True);began=time.monotonic();parts=[]
                for exposure in batch:
                    guard.check(a.lease,300);parts.append(objective.backward(exposure))
                checks=gradients(lm,selected);check_frozen(lm,frozen)
                total=sum(row['weighted_loss'] for row in parts)
                log_observation(active.optimizer,{r['group']+'_objective':r['weighted_loss'] for r in parts},fold='TRAIN')
                rate=active.base.param_groups[0]['lr']
                program=active.step(torch.tensor(total,device='cuda'),number)
                actual=active.optimizer.update
                if actual!=number+1:raise ValueError('Exactly one accumulated native update required')
                check_frozen(lm,frozen);torch.cuda.synchronize();last_seconds=time.monotonic()-began
                row=dict(source=a.source,step=actual,epoch=number//STEPS+1,seconds=last_seconds,
                    LR=rate,objective=total,components=parts,sampler=batch,opal=program,**checks)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(dict(step=actual,objective=total,seconds=last_seconds)),flush=True)
                if actual%8==0:
                    checkpoint(a,folder/'live.pt',lm,active,meta,state['references'],groups)
                if actual%STEPS==0:
                    path=folder/('epoch-%02d.pt'%(actual//STEPS))
                    if path.exists():raise ValueError('Preserve immutable epoch state')
                    value=checkpoint(a,path,lm,active,meta,state['references'],groups)
                    write(folder,'epoch-%02d.json'%(actual//STEPS),dict(source=a.source,
                        epoch=actual//STEPS,lineage_updates=actual,complete323_coverage=True,
                        checkpoint_sha256=value,optimizer_updates_this_epoch=STEPS,
                        behavioral_speech_backend_evaluation_still_required=True,adopted=False))
            write(folder,'training-terminal.json',dict(completed=True,source=a.source,
                actual_new_updates=actual-start,total_lineage_updates=actual,complete_epochs=4,
                final_checkpoint=str(path),checkpoint_sha256=value,final_independent_restore_required=True,
                full_native64_fresh20_speech_backend_live_qualification_required=True,adopted=False))
    except BaseException:
        # Preserve exact actual progress, including steps after the last rolling
        # state. No automatic optimizer retry is performed here.
        value=checkpoint(a,folder/'interrupted.pt',lm,active,meta,state['references'],groups)
        write(folder,'training-interrupted.json',dict(source=a.source,completed=False,
            actual_new_updates=active.optimizer.update-start,total_lineage_updates=active.optimizer.update,
            checkpoint_sha256=value));raise
    finally:signal.signal(signal.SIGTERM,old_handler)


def restore(a,groups,report,folder,*,train=False):
    import torch
    from training.personaplex.matched_state import restore as load_state
    from training.venus.large_phase import write_programs
    from training.venus.state import restore_rng
    guard.check(a.lease,600)
    path=Path(a.checkpoint);expected=a.checkpoint_sha
    if sha(path)!=expected:raise ValueError('Exact actual checkpoint hash required')
    state=torch.load(path,map_location='cpu',weights_only=False)
    lm,loads=model(a,folder);selected=configure(lm,a.capacity)
    programs=write_programs(folder/'programs',state['optimizer']['files'])
    active=runtime(lm,selected,a,folder,programs)
    try:
        step=load_state(state,lm,active,metadata(a,report))
        original=json.loads(Path(str(path)+'.native-reference.json').read_text())
        if (original['checkpoint_sha256']!=expected or not reference_source_matches(a,original['source'])
                or original['execution_id']==os.environ['DUPLEX_NATIVE_EXECUTION_ID']
                or fingerprint(lm,groups)!=original['fingerprints']):
            raise ValueError('Distinct-process actual native restore differs')
        restore_rng(state['rng'])
        write(folder,'restore.json',dict(passed=True,source=a.source,lineage_source=lineage_source(a),base=a.base,capacity=a.capacity,
            data_report_sha256=a.inputs_sha,actual_CUDA=True,
            checkpoint_sha256=expected,reference_execution_id=original['execution_id'],
            execution_id=os.environ['DUPLEX_NATIVE_EXECUTION_ID'],pid=os.getpid(),
            all_Adam_RNG_OPAL_fields_exact=True,native_logits_exact=True,optimizer_updates=0,
            restored_training_step=step,adopted=False))
        if not train:return
        run_training(a,lm,active,state,groups,folder)
    finally:active.close()


def main(a):
    compatibility=validate_lineage(a)
    import torch
    if (a.base not in BASES or a.capacity not in CAPACITIES
            or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=a.source
            or guard.check(a.lease)['source']!=a.source or torch.cuda.device_count()!=1):
        raise ValueError('Exact source-bound learner with one of the job two GPUs required')
    folder=Path(a.root);folder.mkdir(parents=True,exist_ok=False)
    if compatibility:write(folder,'source-compatibility.json',compatibility)
    groups,report=corpus(Path(a.inputs),a.inputs_sha)
    weights_dir=BASE/'public-models/context-spanning-e62a-v1'
    os.environ.update(CS_BASE_DIR=str(BASE/'public-models/personaplex-fdaf-v1'),
        CS_WEIGHTS_DIR=str(weights_dir),CS_VOICES_DIR=str(weights_dir/'voices'))
    sys.path.insert(0,str(BASE/'vendor/context-spanning-79df-v1'))
    from contextspan.moshi.utils.compile import no_compile,no_cuda_graph
    with no_compile(),no_cuda_graph():
        if a.stage=='admit':admit(a,groups,report,folder)
        else:restore(a,groups,report,folder,train=a.stage=='train')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','root','lease','base','capacity','inputs','inputs-sha'):p.add_argument('--'+n,required=True)
    p.add_argument('--lineage-source',choices=('8b822b0d85ccea4f291255204cbc358fa85469c4',))
    p.add_argument('--stage',choices=('admit','restore','train'),required=True)
    for n in ('checkpoint','checkpoint-sha','restore-proof','restore-proof-sha','joint-admission','joint-admission-sha'):
        p.add_argument('--'+n)
    main(p.parse_args())
