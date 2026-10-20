"""Bounded assistant-audio auxiliary stage beside the unchanged actual-history fit.

This is a new eight-update stage from native520, not a continuation of either
rejected joint7530408 or rejected40. All original text/KL/hidden objectives stay.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256

SCHEMA='dl-context-actual-audio49-adapter-proof-v1'
RELOAD_SCHEMA='dl-context-actual-audio49-adapter-reload-v1'


def validate_configuration(args):
    if getattr(args,'native_audio49_stream_gradient_readiness',False):
        from evals.context_native_stream_gradient import validate
        validate(args)
    if getattr(args,'native_audio49_bf16_stream_diagnostic',False):
        from evals.context_native_bf16_stream_diagnostic import validate
        validate(args)
    enabled=bool(getattr(args,'native_audio49_audit',None));weight=getattr(args,'native_audio49_weight',0.)
    smoke=bool(getattr(args,'native_audio49_preflight_only',False));pre=getattr(args,'native_audio49_preflight',None)
    scale=getattr(args,'native_audio49_step_scale',1.)
    if getattr(args,'native_audio49_complete_batch',False) and (not enabled or not getattr(args,'native_audio49_parent_adam',False) or getattr(args,'native_audio49_gradient_probe',False)):raise ValueError('Complete batch is a distinct bounded parent-Adam stage')
    if getattr(args,'native_audio49_parent_adam',False) and (not enabled or scale!=.1 or getattr(args,'native_audio49_gradient_probe',False)):raise ValueError('Parent Adam is one distinct small-step stage, not a gradient probe')
    if getattr(args,'native_audio49_gradient_probe',False) and (not enabled or not smoke or pre or args.reload or args.resume_training):raise ValueError('Gradient probe is zero-update preflight only')
    if type(scale) not in (int,float) or isinstance(scale,bool) or scale not in (1.,.1):raise ValueError('Only the explicit original or tenfold-smaller initial stage is allowed')
    if not enabled:
        if weight or smoke or pre or scale!=1.:raise ValueError('Audio49 options require the complete independent audit')
        return False
    if (weight!=1. or isinstance(weight,bool) or not args.actual_temporal_fit or not args.native_v2_full24_per8
            or args.actual_fit_updates!=8 or args.resume_training or len(args.actual_mixed_data)!=28 or args.native_v2_learning_rate!=2e-6
            or not args.warm_start_proof or Path(args.warm_start_proof).name!='context-training-proof-7510331'
            or args.actual_base_retention_weight!=1. or args.actual_retention_teacher_weight!=1.
            or args.actual_native_teacher_coverage!='all-approved' or args.actual_native_teacher_loss!='balanced-spoken'
            or args.actual_sampling!='balanced-kind' or args.actual_retention_sampling!='balanced-kind'
            or args.native_v2_warm_ordinary_weight!=64 or not args.native_v2_warm_ordinary_clock
            or args.actual_response_first_word_weight!=16. or bool(smoke)==bool(pre)
            or (smoke and args.reload) or args.native_v2_gradient_audit or args.native_v2_retention_drift_proof
            or args.native_v2_learning_curve_audit or args.native_v2_exploration_authorization):
        raise ValueError('Audio49 is one new eight-update native520 stage, with complete old retention and independent BF16 preflight')
    return True


def indices(step):
    if type(step) is not int or not 0<=step<8:raise ValueError('Audio49 has exactly eight initial updates; no continuation here')
    return list(range(step,49,8))



def validate_exposure(steps,teacher_ids,expected_targets):
    if len(steps)!=8 or len(teacher_ids)!=49 or len(set(teacher_ids))!=49 or [s['step'] for s in steps]!=list(range(1,9)):
        raise ValueError('Exact eight-update complete49 schedule required')
    exposures=[];targets=0
    for step,row in enumerate(steps):
        batch=row['assistant_audio49_microbatch'];expected=[teacher_ids[i] for i in indices(step)]
        if ([x['id'] for x in batch]!=expected or any(x['weight']!=8/49
                or x['objective']['user_audio_loss_targets']!=0 or len(x['objective']['assistant_audio_targets'])!=8
                or any(type(n) is not int or n<1 for n in x['objective']['assistant_audio_targets']) for x in batch)):
            raise ValueError('Changed auxiliary sampling or user/codebook target ownership')
        exposures.extend(x['id'] for x in batch);targets+=sum(sum(x['objective']['assistant_audio_targets']) for x in batch)
    if len(exposures)!=49 or len(set(exposures))!=49 or targets!=expected_targets:raise ValueError('Incomplete49 acoustic coverage')
    return True


def implementations():
    return {n:sha256(Path(__file__).with_name(n)) for n in ('context_native_audio49_training.py','context_actual_train.py',
        'context_training_proof.py','context_native_audio49_data.py','context_native_audio49_loss.py')}


def prepare(args,spm,voice):
    validate_configuration(args)
    from evals.context_native_audio49_data import load
    from evals.context_native_audio49_audit import implementations as audit_implementations
    folder=args.native_audio49_audit;gate_file=folder/'audio49-audit.json';gate=json.loads(gate_file.read_text())
    data,batches=load(args.assets,args.out/'voice.pt',spm,voice)
    encoded=json.dumps(data,indent=2)+'\n';digest=hashlib.sha256(encoded.encode()).hexdigest()
    checks=gate['complete49_numerical_cases'];grad=gate['predeclared_longest_kind_gradient_smoke']
    if (gate['schema']!='dl-native-audio49-numerical-gradient-audit-v1' or not gate['passed']
            or gate['optimizer_updates']!=0 or gate['new_checkpoint_written'] or gate['training_approved']
            or not gate['all_model_parameters_unchanged'] or gate['peak_allocated_bytes']>=80_000_000_000
            or gate['settings']['data_sha256']!=digest or sha256(folder/'audio49-data.json')!=digest
            or gate['settings']['implementations']!=audit_implementations() or gate['settings']['threshold']!=.001
            or gate['settings']['reference_kl_weight']!=1. or gate['settings']['reference_hidden_mse_weight']!=10.
            or len(checks)!=49 or [c['id'] for c in checks]!=[b['record']['teacher']['id'] for b in batches]
            or any(not c['passed'] or c['text_alignment_max_error']>.001 or c['causal_prefix_maximum_error']!=0.
                or not c['native_depth_parity']['passed'] for c in checks)
            or len(grad)!=4 or {g['kind'] for g in grad}!={'private','simple','clarify','cancel'}
            or any(not g['finite_temporal_gradients'] or g['nonzero_temporal_gradient_elements']<1
                or g['text_head_audio_gradient_elements']!=0 or g['frozen_parameter_gradients']!=0 for g in grad)):
        raise ValueError('New audio auxiliary lacks complete49 numerical ownership and declared gradient proof')
    policy=dict(mode=('native520-all49-assistant-audio-smallstep-v1' if getattr(args,'native_audio49_step_scale',1.)==.1 else 'native520-all49-assistant-audio-auxiliary-v1'),weight=1.,updates=8,learning_rate=2e-6*getattr(args,'native_audio49_step_scale',1.),
        parent=data['parent'],audit_source=gate['settings']['source_commit'],audit_sha256=sha256(gate_file),
        data_sha256=digest,teacher_ids=[b['record']['teacher']['id'] for b in batches],
        assistant_audio_loss_targets_per8=sum(sum(c['objective']['assistant_audio_targets']) for c in checks),
        original28_data_sha256=data['original28_data_sha256'],required_original_objectives=data['required_original_objectives'],
        original_word64_clock64_and_all_text_losses_unchanged=True,source_audio_words_receipts_folds_unchanged=True,
        assistant_codebooks=list(range(1,9)),user_audio_loss_targets=0,prefix_loss_targets=0,supplied_span_loss_targets=0,
        per_example_weight_each_update=8/49,sampling='Deterministic stride8 partition; all49 exactly once in eight updates; epoch mean weight1',
        implementations=implementations(),new_optimizer=True,extends_rejected_joint_or40=False)
    if getattr(args,'native_audio49_step_scale',1.)!=1.:
        policy.update(step_scale=.1,nominal_legacy_recipe_learning_rate=2e-6,effective_optimizer_learning_rate=2e-7,legacy_objective_rate_field_scope='Original nominal text-recipe identity; combined_recipe.lr and this effective rate govern the optimizer')
    if getattr(args,'native_audio49_parent_adam',False):
        parent_report=json.loads((args.warm_start_proof/'report.json').read_text())
        parent_state=args.warm_start_proof/'training-state.pt'
        if sha256(parent_state)!=parent_report['training_state_sha256']:raise ValueError('Changed original520 optimizer state')
        policy.update(mode='native520-all49-assistant-audio-parent-adam-v1',new_optimizer=False,
            parent_adam_training_state_sha256=sha256(parent_state),parent_adam_parameter_steps=520,
            parent_adam_implementation_sha256=sha256(Path(__file__).with_name('context_native_parent_adam.py')),
            stage_optimizer_updates=8,final_adam_parameter_steps=528,
            stage_update_scope='Eight calls in a new data/recipe stage; original Adam moments and clocks are carried explicitly, original RNG is not imported')
    if getattr(args,'native_audio49_complete_batch',False):
        from evals.context_native_complete_batch import membership
        policy.update(mode='native520-all49-assistant-audio-complete-batch-v1',
            complete_batch_implementation_sha256=sha256(Path(__file__).with_name('context_native_complete_batch.py')),
            complete_batch_membership=membership(data,parent_report),source_unique49_acoustic_targets=policy['assistant_audio_loss_targets_per8'],
            assistant_audio_loss_targets_per8=8*policy['assistant_audio_loss_targets_per8'],per_example_weight_each_update=1/49,
            sampling='Every update includes all28 actual balanced across8 kinds, all4 references, all25 old/all24 new teachers balanced across4 kinds, all49 acoustic mean and all10word/14clock anchors',
            legacy_subpolicy_scope='Original nominal approval/cache/rotation identities; complete_batch_membership governs actual runtime exposure weights. All scalar loss formulas and kind-average masses retained.')
    for name,content in [('audio49-data.json',encoded),('approved-audio49-audit.json',gate_file.read_text())]:
        p=args.out/name
        if args.reload:
            if p.read_text()!=content:raise ValueError('Independent audio49 data/audit reconstruction changed')
        else:p.write_text(content)
    if not args.native_audio49_preflight_only:
        path=args.native_audio49_preflight/'audio49-bf16-preflight.json';pre=json.loads(path.read_text())
        if (pre['schema']!='dl-native-audio49-bf16-preflight-v1' or not pre['passed'] or pre['optimizer_updates']!=0
                or pre['policy']!=policy or pre['source_commit']!=os.environ['DL_TASK_COMMIT']
                or not pre['all_parameters_unchanged'] or len(pre['cases'])!=4):
            raise ValueError('Independent matching BF16 four-kind gradient preflight required')
        if getattr(args,'native_audio49_parent_adam',False):
            if pre.get('parent_adam_import_sha256')!=sha256(args.native_audio49_preflight/'parent-adam-import.json'):raise ValueError('Independent readiness omitted exact parent Adam import')
        if getattr(args,'native_audio49_complete_batch',False):
            complete=args.native_audio49_preflight/'complete-balanced-preflight.json';check=json.loads(complete.read_text())
            if pre.get('complete_balanced_preflight_sha256')!=sha256(complete) or not check['passed'] or check['policy']!=policy or check['optimizer_updates']!=0 or not check['all_parameters_unchanged'] or check['total_components']!=154:raise ValueError('Complete matching154 gradient readiness required')
        policy['bf16_preflight_sha256']=sha256(path)
        target=args.out/'approved-audio49-bf16-preflight.json'
        if args.reload:
            if sha256(target)!=sha256(path):raise ValueError('Packaged BF16 preflight changed')
        else:target.write_bytes(path.read_bytes())
    return policy,batches


def forward(lm,codes):
    from evals.personaplex_depth import native_depth_context
    with native_depth_context(lm):return lm.forward_train(codes)


def fingerprints(lm,batches):
    import torch
    result={}
    with torch.no_grad():
        for row in batches:
            out=forward(lm,row['codes'].cuda());keep=out.mask[:,:8]&row['audio_mask'].cuda()[:,None,:]
            selected=out.logits[:,:8][keep].detach()
            if not selected.numel() or not torch.isfinite(selected).all():raise ValueError('Invalid full49 restore acoustic logits')
            result[row['record']['teacher']['id']]=hashlib.sha256(selected.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            del out,selected
    if len(result)!=49:raise ValueError('Complete49 acoustic restore fingerprints required')
    return result


def backward(lm,batches,step):
    import torch
    from evals.context_native_audio49_loss import loss
    rows=[]
    for index in indices(step):
        row=batches[index];out=forward(lm,row['codes'].cuda());value,detail=loss(out,row['codes'].cuda(),row['audio_mask'].cuda())
        (value*(8/49)).backward()
        rows.append(dict(id=row['record']['teacher']['id'],weight=8/49,acoustic_ce=float(value.detach()),objective=detail))
        del out,value
    return rows


def bf16_preflight(lm,learned,batches,policy,args):
    import torch
    from evals.context_native_audio49_loss import loss
    from evals.personaplex_adapters import checkpoint_language_layers
    from evals.moshi_training_smoke import tensor_hashes
    before=tensor_hashes(lm.named_parameters());rows=[]
    for kind in ['private','simple','clarify','cancel']:
        row=max((r for r in batches if r['record']['teacher']['kind']==kind),key=lambda r:r['codes'].shape[-1])
        for p in learned.values():p.grad=None
        with checkpoint_language_layers(lm):
            out=forward(lm,row['codes'].cuda());value,detail=loss(out,row['codes'].cuda(),row['audio_mask'].cuda());value.backward()
        temporal={n:p for n,p in learned.items() if not n.startswith('text_linear.')}
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in temporal.values()):raise ValueError('Invalid BF16 acoustic gradient')
        if any(p.grad is not None for n,p in lm.named_parameters() if n not in learned):raise ValueError('Frozen parameter acquired BF16 acoustic gradient')
        if any(p.grad is not None and p.grad.count_nonzero() for n,p in learned.items() if n.startswith('text_linear.')):raise ValueError('Acoustic loss unexpectedly supervises text head')
        count=sum(int(p.grad.count_nonzero()) for p in temporal.values())
        if not count:raise ValueError('Dead BF16 acoustic gradient')
        rows.append(dict(id=row['record']['teacher']['id'],kind=kind,acoustic_ce=float(value.detach()),nonzero_temporal_gradient_elements=count,objective=detail))
        for p in learned.values():p.grad=None
        del out,value
    unchanged=before==tensor_hashes(lm.named_parameters())
    peak=torch.cuda.max_memory_allocated()
    result=dict(schema='dl-native-audio49-bf16-preflight-v1',passed=unchanged and peak<80_000_000_000,policy=policy,cases=rows,peak_allocated_bytes=peak,
        all_parameters_unchanged=unchanged,optimizer_updates=0,source_commit=os.environ['DL_TASK_COMMIT'],pid=os.getpid(),
        model_dtype='BF16 base plus FP32 temporal adapters, MATH attention',training_approved=False)
    if getattr(args,'native_audio49_parent_adam',False):result['parent_adam_import_sha256']=sha256(args.out/'parent-adam-import.json')
    (args.out/'audio49-bf16-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise ValueError('BF16 preflight mutated weights or exceeded80GB')
    return result


def committed_implementations(commit):
    if not isinstance(commit,str) or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Exact immutable builder commit required')
    return {name:hashlib.sha256(subprocess.check_output(['git','show',commit+':evals/'+name],cwd=str(Path(__file__).resolve().parent))).hexdigest() for name in implementations()}


def verify_checkpoint(proof):
    from evals.context_rollout_conditioning import verify_checkpoint as old_verify
    from evals.context_spanning_download import SOURCE_COMMIT
    proof=Path(proof);r=json.loads((proof/'report.json').read_text());v=json.loads((proof/'reload-report.json').read_text())
    p=r.get('assistant_audio_auxiliary',{});data=json.loads((proof/'data.json').read_text())
    if (r['schema']!=SCHEMA or not r['passed'] or not r['frozen_weights_unchanged'] or not r['base_audio_parameters_frozen']
            or r['source_commit']!=SOURCE_COMMIT or r['optimizer_steps']!=8 or r['initial_optimizer_steps']!=0
            or r['updates_this_run']!=8 or r['warm_start_optimizer_reset']!=(p.get('mode') not in ('native520-all49-assistant-audio-parent-adam-v1','native520-all49-assistant-audio-complete-batch-v1')) or r['resumed_from_training_state']
            or r['last_layers']!=4 or r['rank']!=8 or r['scaling']!=2. or r['past_loss_targets']!=0
            or r['audio_loss_targets']<1 or r['original_actual_history_audio_loss_targets']!=0
            or r['user_audio_loss_targets']!=0 or r['prefix_audio_loss_targets']!=0 or r['supplied_span_audio_loss_targets']!=0
            or not r['all_actual_temporal_audits_passed'] or not r['all_component_audits_passed']
            or not r['zero_adapter_native_batch_logits_exact'] or len(data['components'])!=28
            or data['audio_loss_targets']!=0 or data['past_loss_targets']!=0 or not data['all_component_audits_passed']
            or p.get('mode') not in ('native520-all49-assistant-audio-auxiliary-v1','native520-all49-assistant-audio-smallstep-v1','native520-all49-assistant-audio-parent-adam-v1','native520-all49-assistant-audio-complete-batch-v1') or p['weight']!=1. or p['updates']!=8
            or len(p['teacher_ids'])!=49 or len(set(p['teacher_ids']))!=49
            or p['required_original_objectives']!=dict(actual_histories=28,old_native_teachers=25,new_native_teachers=24,kl_weight=1.,hidden_mse_weight=10.)
            or p['implementations']!=committed_implementations(r['repo_commit']) or not p['original_word64_clock64_and_all_text_losses_unchanged']
            or len(r['retention_teacher_text']['teachers'])!=25 or len(r['additional_native_v2']['teacher']['teachers'])!=24
            or r['warm_ordinary_retention']['weight']!=64. or r['warm_ordinary_clock_retention']['weight']!=64.
            or v['schema']!=RELOAD_SCHEMA or not v['passed'] or not v['fresh_process'] or v['pid']==v['training_pid']
            or not v['all_valid_logits_exact'] or not v['all49_auxiliary_logits_exact'] or not v['optimizer_rng_state_restored']
            or r['combined_recipe']['assistant_audio_auxiliary']!=p
            or r['combined_recipe']['retention_kl_weight']!=1. or r['combined_recipe']['retention_hidden_mse_weight']!=10.
            or r['combined_recipe']['actual_component_count']!=28
            or not v['optimizer_and_rng_independently_compared_exact'] or v['actual']!=r['expected_logits']
            or v['actual_audio49_logits']!=r['expected_audio49_logits']):
        raise ValueError('Unverified bounded audio49 combined-objective checkpoint')
    expected_scale=.1 if p['mode'] in ('native520-all49-assistant-audio-smallstep-v1','native520-all49-assistant-audio-parent-adam-v1','native520-all49-assistant-audio-complete-batch-v1') else 1.
    if (p.get('step_scale',1.)!=expected_scale or p['learning_rate']!=2e-6*expected_scale
            or r['combined_recipe']['lr']!=p['learning_rate']
            or r['additional_native_v2']['learning_rate']!=2e-6):raise ValueError('Changed declared effective optimizer rate or original text-recipe identity')
    if expected_scale==.1 and (p['nominal_legacy_recipe_learning_rate']!=2e-6 or p['effective_optimizer_learning_rate']!=2e-7):raise ValueError('Smallstep optimizer identity changed')
    if p['mode'] in ('native520-all49-assistant-audio-parent-adam-v1','native520-all49-assistant-audio-complete-batch-v1'):
        expected=hashlib.sha256(subprocess.check_output(['git','show',r['repo_commit']+':evals/context_native_parent_adam.py'],cwd=str(Path(__file__).resolve().parent))).hexdigest()
        if p['new_optimizer'] or p['parent_adam_implementation_sha256']!=expected or p['parent_adam_parameter_steps']!=520 or p['final_adam_parameter_steps']!=528:raise ValueError('Changed Adam transplant implementation or clocks')
        from evals.context_native_parent_adam import validate_finished
        validate_finished(proof,r)
    parent_identity,parent=old_verify(Path(p['parent']['proof']))
    if p['mode'] in ('native520-all49-assistant-audio-parent-adam-v1','native520-all49-assistant-audio-complete-batch-v1') and p['parent_adam_training_state_sha256']!=parent['training_state_sha256']:raise ValueError('Adam source is not the approved native520 parent')
    if (parent_identity!=p['parent'] or r['warm_start']!=parent_identity or parent['optimizer_steps']!=520
            or r['data_sha256']!=parent['data_sha256'] or r['native_protocol']!=parent['native_protocol']
            or r['frozen_hashes']!=parent['frozen_hashes'] or r['voice_sha256']!=parent['voice_sha256']
            or r['voice_reference_sha256']!=parent['voice_reference_sha256']
            or r['assets_manifest_sha256']!=parent['assets_manifest_sha256']
            or r['retention_examples']!=parent['retention_examples']
            or r['actual_temporal_audit_sha256']!=parent['actual_temporal_audit_sha256']):
        raise ValueError('Audio49 lost exact native520 parent/original28 data')
    for filename,key in [('adapters.safetensors','checkpoint_sha256'),('training-state.pt','training_state_sha256'),
            ('data.json','data_sha256'),('voice.pt','voice_sha256'),('approved-temporal-audit.json','actual_temporal_audit_sha256')]:
        if sha256(proof/filename)!=r[key]:raise ValueError('Changed audio49 checkpoint artifact')
    if (sha256(proof/'approved-audio49-bf16-preflight.json')!=p['bf16_preflight_sha256']
            or sha256(proof/'audio49-data.json')!=p['data_sha256'] or sha256(proof/'approved-audio49-audit.json')!=p['audit_sha256']
            or v['checkpoint_sha256']!=r['checkpoint_sha256'] or v['training_state_sha256']!=r['training_state_sha256']
            or v['data_sha256']!=r['data_sha256'] or len(r['expected_audio49_logits'])!=49):raise ValueError('Changed audio49 proof identities')
    if p['mode']=='native520-all49-assistant-audio-complete-batch-v1':
        from evals.context_native_complete_batch import membership,validate_exposure as validate_complete
        expected=hashlib.sha256(subprocess.check_output(['git','show',r['repo_commit']+':evals/context_native_complete_batch.py'],cwd=str(Path(__file__).resolve().parent))).hexdigest()
        package=json.loads((proof/'audio49-data.json').read_text());members=membership(package,parent)
        audited=json.loads((proof/'approved-audio49-audit.json').read_text())
        unique_targets=sum(sum(x['objective']['assistant_audio_targets']) for x in audited['complete49_numerical_cases'])
        if p['source_unique49_acoustic_targets']!=unique_targets:raise ValueError('Complete-batch source target count differs from all49 independent numerical cases')
        if p['complete_batch_implementation_sha256']!=expected or p['complete_batch_membership']!=members or p['assistant_audio_loss_targets_per8']!=8*p['source_unique49_acoustic_targets']:raise ValueError('Changed complete-batch implementation or mass')
        validate_complete(r['steps'],p['teacher_ids'],p['source_unique49_acoustic_targets'],members)
    else:validate_exposure(r['steps'],p['teacher_ids'],p['assistant_audio_loss_targets_per8'])
    if set(r['expected_audio49_logits'])!=set(p['teacher_ids']) or r['audio_loss_targets']!=p['assistant_audio_loss_targets_per8']:
        raise ValueError('Changed full49 restore or target identity')
    identity=dict(proof=str(proof),report_sha256=sha256(proof/'report.json'),reload_report_sha256=sha256(proof/'reload-report.json'),
        checkpoint_sha256=r['checkpoint_sha256'],training_state_sha256=r['training_state_sha256'],native_protocol=r['native_protocol'],optimizer_steps=8)
    return identity,r
