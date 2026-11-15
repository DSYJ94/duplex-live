"""One coherent expanded native SFT phase; fresh state proof before learning.

Existing196-update studies are never resumed/replayed here. This NEW objective
adds all1008 qualified TRAIN pairs while retaining every old objective/weight.
"""
import argparse,json,os,time,uuid,math,shutil,re
from pathlib import Path
from types import SimpleNamespace

OLD_REPORT_SHA='72e2b514b1c922f6591e18e1d30b198628170db2a2cc5e2efdb12d8d3d538e26'
OLD_SHA='a49375937dd8377ed6098011f91bb729fb6843939373fa4cc922d819d7c12efa'
BASE=Path('/projects/u6ada/duplex-live')


def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def run(a):
    adaptive_stages={'transition','verify-transition','train-adaptive'}
    if a.stage in adaptive_stages and not getattr(a,'adaptive_plan',None):raise ValueError('Adaptive stage requires an explicit safe handover plan')
    if getattr(a,'adaptive_plan',None) and a.stage not in adaptive_stages|{'verify'}:raise ValueError('Use the explicit adaptive restore/train stages')
    import torch,torch.distributed as dist
    from datetime import timedelta
    from evals.moshi_model_download import sha256 as sha
    from evals.context_native_audio49_data import digest
    from training.venus.state import rng_state,restore_rng,exact_state
    from training.venus.native_optimizer import assert_optimizer_equal
    from training.venus.opal_adapter import log_observation
    from .expanded_guard import check,AUTH
    from .matched_native import model,corpus,fingerprint,gradients,frozen_versions,check_frozen,RECIPE,worst_rows
    from .matched_state import materialize_adam
    from .matched_schedule import epoch
    from .expanded_schedule import build
    from .expanded_provenance import capture_provenance
    from .distributed_native import DistributedNative,cpu_state
    from .distributed_opal import DistributedOpal
    from .distributed_objective import backward_retained
    from .distributed_expanded_objective import backward as new_backward
    from .distributed_train import update
    from .distributed_state import save,restore
    adaptive_plan = json.loads(a.adaptive_plan.read_text()) if getattr(a,'adaptive_plan',None) else None
    adaptive = adaptive_plan is not None
    if adaptive:
        from .adaptive_handover import restore_identity,target_identity,native_restore_packet,all_native_restored
        from .adaptive_native import context as boundary_context,request_for,publish_decision
        if str(a.root.resolve())!=adaptive_plan['root']or str(a.backup.resolve())!=adaptive_plan['backup']or a.source!=adaptive_plan['source']:
            raise ValueError('Exact adaptive output/source/backup plan required')
        if a.stage=='admit':raise ValueError('Reuse proven admission; do not repeat calibration')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=4 or torch.cuda.device_count()!=4:raise ValueError('Exactly four independently allocated native GPUs required')
    torch.cuda.set_device(local);dist.init_process_group('nccl',timeout=timedelta(minutes=30));check(a.lease,600)
    execution=uuid.uuid4().hex;folder=a.root/('job-'+os.environ['SLURM_JOB_ID'])/('train' if adaptive and a.stage=='train-adaptive' else a.stage)/('rank-%02d'%rank);folder.mkdir(parents=True,exist_ok=False)
    native=json.loads((a.expanded/'inputs.json').read_text())
    if not native['complete1072'] or native['TRAIN']!=1008 or native['DEV']!=64 or len(native['rows'])!=1072 or len({r['id']for r in native['rows']})!=1072:raise ValueError('All independently reconstructed new native data required')
    provenance=capture_provenance(native,a.source)
    table={r['id']:r for r in native['rows']};loaded={}
    def item(ident):
        if ident not in loaded:
            r=table[ident];path=Path(r['inputs_file'])
            if not r['native_qualification'] or sha(path)!=r['inputs_sha256']:raise ValueError('Native source identity changed')
            s=torch.load(path,map_location='cpu',weights_only=True)
            loaded[ident]=dict(id=ident,fold=r['fold'],inputs=s['inputs'],targets=s['targets'],mask=s['mask'],codes=s['codes'][None],audio_mask=s['mask'][None],native_qualification=True)
        return loaded[ident]
    groups,old=corpus(a.old_inputs,OLD_REPORT_SHA)
    if old['input_sha256']!=OLD_SHA:raise ValueError('Exact original tensor artifact required')
    schedule=build(groups,native['rows'],world_size=world);flat=[x for e in schedule['schedule']for x in e]
    recipe=dict(RECIPE,updates=196,epochs=4,new_TRAIN=1008,new_DEV=64,new_objective='equal-example cycle mean text+all8 native assistant acoustic heads',native_world_size=4,checkpoint_cadence='every completed update; immutable epoch states; rolling GC only after backup')
    lm,loads=model(SimpleNamespace(base='personaplex'),folder);baseline=fingerprint(lm,groups);refs={};repeated={}
    with torch.no_grad():
        for r in groups['reference']:
            m=r['mask'].cuda();h,z=lm.forward_codes(r['inputs'].cuda());repeated[r['id']]=dict(hidden=digest(h[:,m]),text_logits=digest(z[0,0,m]));refs[r['id']]=(h[:,m].detach().cpu().clone(),z[0,0,m].float().softmax(-1).detach().cpu())
    write(folder/'zero-raw.json',dict(source=a.source,baseline=baseline,repeated=repeated,exact=baseline==repeated))
    if repeated!=baseline:raise ValueError('Raw native repeat is not exact before adapter insertion')
    def configured(model,ownership):
        actual=fingerprint(model,groups)
        write(folder/'zero-configured.json',dict(source=a.source,ownership=ownership,baseline=baseline,actual=actual,exact=actual==baseline))
        if ownership['trainable_parameters']!=105070592:raise ValueError('Exact broad parameter ownership failed')
        if actual!=baseline:raise ValueError('Adapter insertion changed native outputs before distributed wrapping')
    runtime=DistributedNative(lm,'broad-lora64',torch.device('cuda',local),after_configure=configured);selected={n:p for n,p in lm.named_parameters()if p.requires_grad}
    wrapped=fingerprint(runtime.view,groups)
    write(folder/'zero-wrapped.json',dict(source=a.source,baseline=baseline,actual=wrapped,exact=wrapped==baseline))
    if wrapped!=baseline:raise ValueError('Distributed wrapping changed native outputs')
    base=torch.optim.AdamW(runtime.parameters(),lr=recipe['learning_rate'],betas=tuple(recipe['adam_betas']),eps=recipe['adam_eps'],weight_decay=recipe['weight_decay'],foreach=False,fused=False,amsgrad=False)
    adam=materialize_adam(base);opal=DistributedOpal(runtime,base,source=BASE/'opal-upstream-v1',run_dir=folder/'opal',recipe=recipe,adaptive=adaptive)
    identity=dict(source=a.source,base='personaplex',method='broad-lora64',trainable_parameters=105070592,world=world,old_inputs_sha256=OLD_SHA,old_input_report_sha256=OLD_REPORT_SHA,capture_provenance=provenance,expanded_inputs_sha256=sha(a.expanded/'inputs.json'),authorization_sha256=sha(AUTH),recipe=recipe,native_data_qualified=True,new_experiment=True,old_saved_updates_replayed=0)
    checkpoints=a.root/'checkpoints'
    if rank==0:checkpoints.mkdir(exist_ok=True)
    dist.barrier();frozen=frozen_versions(lm)
    def checkpoint(step,name=None):
        path=checkpoints/(name or ('step-%04d.pt'%step));base.zero_grad(set_to_none=True);rng=rng_state();fp=fingerprint(runtime.view,groups);restore_rng(rng)
        state=opal.rank_state(rng);state.update(native_fingerprint=fp,writer_execution=execution,writer_pid=os.getpid(),new_step=step)
        save(path,runtime,base,identity=identity,step=step,rank_state=state)
        if rank==0:write(Path(str(path)+'.sha256.json'),dict(file=path.name,sha256=sha(path),step=step,source=a.source))
        dist.barrier();return path
    def backup(path):
        if rank==0:
            saved_step=torch.load(path,map_location='cpu',weights_only=False)['step'];target=a.backup/(path.stem if path.stem.startswith(('adaptive-','transition-'))else 'step-%04d'%saved_step);target.mkdir(parents=True,exist_ok=True);files=[path,Path(str(path)+'.sha256.json')]+sorted(path.parent.glob(path.name+'.rank-*.pt'));receipts=[]
            for source in files:
                dest=target/source.name
                if not dest.exists():shutil.copyfile(source,dest)
                with dest.open('r+b')as f:f.flush();os.fsync(f.fileno())
                value=sha(source)
                if sha(dest)!=value:raise ValueError('Independent checkpoint backup differs')
                receipts.append(dict(file=dest.name,sha256=value,bytes=dest.stat().st_size))
            receipt=dict(complete=True,source=str(path),files=receipts,step=saved_step)
            if (target/'backup.json').exists():
                if json.loads((target/'backup.json').read_text())!=receipt:raise ValueError('Immutable backup identity changed')
            else:write(target/'backup.json',receipt)
            # Only redundant states of THIS new phase are collected. All old
            # studies, initialization, immutable epochs and latest two states stay.
            current=saved_step;keep={0,current,current-1}|{n for n in range(49,current+1,49)}
            for old_dir in a.backup.glob('step-*'):
                n=int(old_dir.name[5:])
                if n not in keep and n<current and (old_dir/'backup.json').exists():shutil.rmtree(old_dir)
            for old_file in checkpoints.glob('step-*.pt'):
                if not re.fullmatch(r'step-\d{4}\.pt',old_file.name):continue
                n=int(old_file.stem[5:])
                if n not in keep and n<current:
                    for leaf in [old_file,Path(str(old_file)+'.sha256.json')]+list(checkpoints.glob(old_file.name+'.rank-*.pt')):
                        if leaf.exists():leaf.unlink()
        dist.barrier();return a.backup/(path.stem if path.stem.startswith(('adaptive-','transition-'))else 'step-%04d'%(0 if path.stem=='init' else int(path.stem[5:])))/'backup.json'
    def development(step):
        from evals.context_native_stream_gradient import coefficients
        from evals.context_native_audio49_training import forward as audio_forward
        from evals.context_native_audio49_loss import loss as audio_loss
        from torch.nn import functional as F
        results=[];dev=[r for r in native['rows']if r['fold']=='DEV']
        with torch.no_grad():
            for r in dev[rank::world]:
                check(a.lease,120);row=item(r['id']);m=row['mask'].cuda();h,z=runtime.view.forward_codes(row['inputs'].cuda());y=row['targets'].cuda();text=(F.cross_entropy(z[0,0,m].float(),y[m],reduction='none')*coefficients(y,m)[m]).sum();audio,_=audio_loss(audio_forward(runtime.view,row['codes'].cuda()),row['codes'].cuda(),row['audio_mask'].cuda());results.append(dict(id=r['id'],text=float(text),audio=float(audio)))
        all_rows=[None]*world;dist.all_gather_object(all_rows,results);values=[r for batch in all_rows for r in batch]
        if len(values)!=64 or any(not math.isfinite(r[k])for r in values for k in ('text','audio')):raise ValueError('Complete finite native64 DEV required')
        metrics={k:sum(r[k]for r in values)/64 for k in ('text','audio')};opal.active.development(metrics)
        if rank==0:write(a.root/('dev-step-%04d.json'%step),dict(step=step,rows=values,mean=metrics,free_generation_behavior_qualified=False))
    try:
        if a.stage=='admit':
            before={n:digest(p)for n,p in selected.items()};chosen=worst_rows(groups);exposures=[next(x for batch in epoch(groups,0)for x in batch if x['group']==g and x['index']==i)for g,i in chosen.items()];base.zero_grad(set_to_none=True);began=time.monotonic();old_loss=backward_retained(runtime,groups,refs,exposures);torch.cuda.synchronize();old_seconds=time.monotonic()-began;old_gradient=gradients(lm,selected)
            base.zero_grad(set_to_none=True);worst=max((r for r in native['rows']if r['fold']=='TRAIN'),key=lambda r:r['ownership']['full_real_frames']+r['ownership']['prefix_frames']);began=time.monotonic();new_loss=new_backward(runtime,item(worst['id']),valid=True,coefficient=1.,global_batch_count=world);torch.cuda.synchronize();new_seconds=time.monotonic()-began;new_gradient=gradients(lm,selected)
            if before!={n:digest(p)for n,p in selected.items()}:raise ValueError('Admission changed trainable weights')
            check_frozen(lm,frozen);base.zero_grad(set_to_none=True);path=checkpoint(0,'init.pt');parts=[None]*world;dist.all_gather_object(parts,dict(rank=rank,pid=os.getpid(),execution=execution,old_gradient=old_gradient,new_gradient=new_gradient,old_worst_seconds=old_seconds,new_worst_seconds=new_seconds,materialized_Adam=adam,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            if rank==0:write(a.root/'admission.json',dict(source=a.source,actual_CUDA=True,all_four_ranks=parts,zero_expansion_native_exact=True,actual_new_updates=0,checkpoint=str(path),checkpoint_sha256=sha(path),worst_update_seconds=max(r['old_worst_seconds']+6*r['new_worst_seconds']for r in parts),total_new_updates=196,complete_epochs=4,independent_fresh_restore_pending=True))
            lease=json.loads(a.lease.read_text());size=sum(f.stat().st_size for f in [path]+list(path.parent.glob(path.name+'.rank-*.pt')))
            if size*8*1.1>lease['home_hard_bytes']-lease['home_used_bytes']:raise ValueError('Complete phase checkpoint retention does not fit actual user quota')
            backup(path);return
        if a.checkpoint is None:raise ValueError('Train requires an actual exact saved state')
        if adaptive:
            error=None
            if rank==0:
                try:
                    metadata=json.loads(Path(str(a.checkpoint)+'.sha256.json').read_text())
                    if sha(a.checkpoint)!=metadata['sha256']:raise ValueError('Common native checkpoint bytes differ')
                    if a.stage=='transition'and metadata['sha256']!=adaptive_plan['parent_checkpoint_sha256']:raise ValueError('Wrong parent checkpoint')
                except Exception as exc:error=repr(exc)
            errors=[error];dist.broadcast_object_list(errors,src=0)
            if errors[0]:raise ValueError(errors[0])
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);expected_rank={}
        original_identity=identity
        expected_identity=identity
        if adaptive:
            if a.stage=='transition':expected_identity=restore_identity(saved['identity'],identity,adaptive_plan)
            else:expected_identity=target_identity(identity,adaptive_plan)

        def rank_restore(value):
            if value['writer_execution']==execution:raise ValueError('Fresh process required')
            opal.active.optimizer.load_state_dict(value['opal']);opal.active.restore_diagnostics(value['diagnostics']);opal.restore_controller(value);restore_rng(value['rng']);assert_optimizer_equal(value['opal'],opal.active.optimizer.state_dict())
            if not exact_state(value['rng'],rng_state()) or not exact_state(value['recipe'],recipe) or not exact_state(value['diagnostics'],opal.active.diagnostics_state):raise ValueError('Exact RNG/recipe/diagnostics restore required')
            if fingerprint(runtime.view,groups)!=value['native_fingerprint']:raise ValueError('Fresh native fingerprints differ')
            restore_rng(value['rng']);expected_rank.update(value)
        step=restore(saved,runtime,base,identity=expected_identity,restore_rank_state=rank_restore,checkpoint_directory=a.checkpoint.parent)
        if not exact_state(saved['training']['optimizer'],cpu_state(base.state_dict())):raise ValueError('Every common Adam field must restore exactly')
        if adaptive:
            packet=native_restore_packet(source=a.source,step=step,execution=execution,pid=os.getpid(),rank=rank,value=expected_rank,
                optimizer_exact=True,rng_exact=exact_state(expected_rank['rng'],rng_state()),recipe_exact=exact_state(expected_rank['recipe'],recipe),
                diagnostics_exact=exact_state(expected_rank['diagnostics'],opal.active.diagnostics_state),fingerprint_exact=True,
                common_adam_exact=exact_state(saved['training']['optimizer'],cpu_state(base.state_dict())))
            identity=target_identity(original_identity,adaptive_plan)
        else:packet=dict(native_data_qualified=True,native_fresh_restore_passed=True,source=a.source,actual_CUDA=True,step=step,execution=execution,reference_execution=expected_rank['writer_execution'],pid=os.getpid(),all_Adam_RNG_OPAL_native_fields_exact=True)
        write(folder/'restore.json',packet)
        proofs=[None]*world;dist.all_gather_object(proofs,packet)
        if adaptive:all_native_restored(proofs,step)
        if rank==0:write(folder.parent/'restore-all-ranks.json',dict(passed=True,rows=proofs))
        dist.barrier()
        if adaptive and a.stage=='verify-transition':
            if not opal.boundary.state['enabled']:raise ValueError('Restored controller not enabled')
            if rank==0:write(folder.parent/'verify-terminal.json',dict(passed=True,step=step,new_updates=0,checkpoint_sha256=sha(a.checkpoint)))
            return
        def adaptive_boundary(path,receipt):
            base.zero_grad(set_to_none=True)
            values=None
            if rank==0:
                try:
                    values=dict(context=boundary_context(path,receipt,a.expanded/'inputs.json'),request=request_for(adaptive_plan,a.root,a.expanded,a.old_inputs,receipt,step,opal.boundary.state['current_multiplier']),error=None)
                except Exception as exc:values=dict(error=repr(exc))
            packets=[values];dist.broadcast_object_list(packets,src=0);values=packets[0]
            if values['error']:raise ValueError('Complete native decision evidence unavailable: '+values['error'])
            return publish_decision(opal.boundary,checkpoint=checkpoint,backup_checkpoint=backup,context_value=values['context'],
                request=values['request'] if rank==0 else None,step=step,
                write_receipt=lambda value:write(folder.parent/('adaptive-decision-%04d.json'%step),value)if rank==0 else None)
        if adaptive and a.stage=='transition':
            if step!=adaptive_plan['step']:raise ValueError('Exact latest cooperative parent step required')
            last=adaptive_boundary(a.checkpoint,Path(adaptive_plan['parent_backup_receipt']))
            if rank==0:write(folder.parent/'transition-terminal.json',dict(passed=True,step=step,new_updates=0,checkpoint=str(last),checkpoint_sha256=sha(last)))
            return
        if a.stage=='verify':
            if step!=196:raise ValueError('Final verification requires all196 new updates')
            if rank==0:write(folder.parent/'verify-terminal.json',dict(passed=True,step=step,new_updates=0))
            return
        admission=json.loads(((Path(adaptive_plan['parent_root'])if adaptive else a.root)/'admission.json').read_text());worst=admission['worst_update_seconds'];last=a.checkpoint
        if step==0 and not (a.root/'dev-step-0000.json').exists():development(0)
        for index in range(step,len(flat)):
            try:check(a.lease,math.ceil(worst*1.25)+120)
            except TimeoutError:break
            row=flat[index];slots=[dict(row=item(slot[rank]['id']),valid=slot[rank]['valid'])for slot in row['microbatches']];lr=base.param_groups[0]['lr'];began=time.monotonic()
            result=update(runtime,base,opal,step=index,retained_groups=groups,references=refs,exposures=row['retained'],expanded_row=None,valid=True,expanded_coefficient=row['new_coefficient'],identity=identity,checkpoint_path=None,log_path=folder/'training.jsonl',expanded_microbatches=slots,admission=packet)
            used_lr=result['opal'].get('effective_learning_rates',[lr])[0]
            log_observation(opal.active.optimizer,dict(native_total=result['global_loss'],native_learning_rate=used_lr,completed_new_updates=index+1),fold='TRAIN');check_frozen(lm,frozen)
            if (index+1)%49==0:development(index+1)
            last=checkpoint(index+1)
            if (index+1)%8==0 or (index+1)%49==0:
                receipt=backup(last)
                if adaptive and (index+1)%49==0:
                    step=index+1;last=adaptive_boundary(last,receipt)
            if rank==0:print(json.dumps(dict(event='actual_new_optimizer_update',step=index+1,total=196,seconds=time.monotonic()-began,learning_rate_used=used_lr,checkpoint=str(last))),flush=True)
            step=index+1
        if step%8 and step%49:backup(last)
        if rank==0:write(folder.parent/'terminal.json',dict(source=a.source,actual_new_step=step,training_complete=step==196,continuation_ready=0<step<196,checkpoint=str(last),checkpoint_sha256=sha(last),native_behavior_speech_backend_live_acceptance_pending=True))
    finally:
        opal.close();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['admit','train','verify','transition','verify-transition','train-adaptive'])
    for n in ('root','expanded','old-inputs','lease','backup'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--adaptive-plan',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--source',required=True);run(p.parse_args())
