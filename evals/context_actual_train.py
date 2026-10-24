"""Bounded resumable actual-history adaptation with independent TRAIN retention."""
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from evals.moshi_model_download import sha256


def run(lm, spm, voice, examples, components, audits, args):
    import torch
    from torch.nn import functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from safetensors.torch import save_file, load_file
    from contextspan.moshi.models.lm import _delay_sequence
    from contextspan.moshi.utils.compile import no_compile, no_cuda_graph
    from evals.personaplex_adapters import install_adapters, adapter_parameters, checkpoint_language_layers
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_spanning_download import SOURCE_COMMIT
    from evals.context_training_state import save as save_state, restore as restore_state
    from evals.context_native_v2_teacher_objective import learning_rate
    rate = learning_rate(args)*getattr(args,'native_audio49_step_scale',1.)
    source_transition=None
    if getattr(args,'native_v2_exploration_authorization',None):
        from evals.context_native_exploration import validate as validate_exploration
        source_transition=validate_exploration(args)['source_transition']
    pre = args.actual_temporal_preflight
    gate = json.loads((pre/'actual-temporal-audit.json').read_text())
    if (gate['schema'] != 'dl-context-actual-temporal-audit-v1' or gate['passed'] is not True
            or gate['data_sha256'] != sha256(args.out/'data.json') or gate['optimizer_steps'] != 0
            or len(gate['components']) != len(audits)):
        raise RuntimeError('Actual temporal fit requires the passing matching zero-update audit')
    for checked, audit in zip(gate['components'],audits):
        if (any(checked.get(k) != v for k,v in audit.items()) or checked['passed'] is not True
                or checked['causal_prefix_maximum_error'] != 0.
                or set(checked['comparisons']) != {'hidden','text_logits'}
                or any(not 0. <= v['maximum_absolute_error'] <= .001 for v in checked['comparisons'].values())):
            raise RuntimeError('Actual temporal audit changed or failed')
    if not torch.equal(voice,torch.load(pre/'voice.pt',map_location='cpu',weights_only=True)['codes']):
        raise RuntimeError('Temporal preflight used different voice codes')
    embedded = args.out/'approved-temporal-audit.json'
    if args.reload:
        if sha256(embedded) != sha256(pre/'actual-temporal-audit.json'):
            raise RuntimeError('Changed packaged temporal audit')
    else:
        embedded.write_bytes((pre/'actual-temporal-audit.json').read_bytes())
    batches = [];component_kinds=[];response_boundaries=[]
    from evals.context_decision_loss import weights as text_weights,policy as decision_policy
    warm_start=None
    if args.warm_start_proof:
        from evals.context_rollout_conditioning import verify_checkpoint
        warm_start,parent=verify_checkpoint(args.warm_start_proof)
        if (parent['assets_manifest_sha256']!=sha256(args.assets)
                or parent['voice_reference_sha256']!=sha256(args.voice_reference)
                or not torch.equal(voice,torch.load(args.warm_start_proof/'voice.pt',map_location='cpu',weights_only=True)['codes'])):
            raise RuntimeError('Warm-start source assets or voice differ from the new corrective stage')
    for folder, component in zip(args.actual_head_data,components):
        from evals.context_component_kind import manifest_name
        name=manifest_name(folder,early=bool(args.actual_early_data),mixed=bool(args.actual_mixed_data))
        m = json.loads((folder/name).read_text()); h = Path(m['actual_history'])
        response_boundaries.append(m.get('native_response_boundary_version') in (2,3)
            and (m.get('target_kind')=='ordinary_answer' or (m.get('native_response_boundary_version')==3
                and m.get('target_kind')=='clarification_question' and m.get('question_retiming_before_project_verified') is True))
            and m.get('actual_boundary_cache_verified') is True)
        component_kinds.append(m.get('target_kind','private_result'))
        record = json.loads((h/'student-history.json').read_text())
        raw = torch.load(h/'student-history.pt',map_location='cpu',weights_only=True)['delayed_inputs']
        live = [c for c in record['calls'] if c['mode']=='live']
        stop = live[m['future_first_real_frame']]['start']
        inputs = torch.cat([raw[:,:,:stop],component['delayed_inputs']],dim=2).cuda()
        mask = torch.zeros(inputs.shape[-1],dtype=torch.bool,device='cuda')
        mask[stop:] = component['text_mask'].cuda()
        target = torch.zeros(inputs.shape[-1],dtype=torch.int64,device='cuda')
        target[stop:] = component['text_targets'].cuda()
        if mask[:stop].any() or int(mask.sum()) != int(component['text_mask'].sum()):
            raise RuntimeError('Actual past accidentally acquired labels')
        batches.append((inputs,mask,target))
    decision=decision_policy(args.actual_response_first_word_weight,response_boundaries)
    balance_retention=args.actual_retention_sampling=='balanced-kind'
    retention = [];retention_targets=[];retention_teacher_audits=[]
    teacher_weight=args.actual_retention_teacher_weight
    native_teacher_mode=bool(getattr(args,'actual_native_teacher_preflight',None));native_teacher_batches=[]
    from evals.context_native_teacher_objective import validate_base_retention
    base_retention_weight=validate_base_retention(args.actual_base_retention_weight,native_teacher_mode)
    for kind in ('private','simple','clarify','cancel'):
        e = max((e for e in examples if e['record']['kind']==kind),key=lambda e:e['codes'].shape[-1])
        codes = e['codes'][None].cuda(); initial = lm._get_initial_token()
        inputs = torch.cat([initial,_delay_sequence(lm.delays,codes,initial)],dim=2)[:,:,:-1]
        retention.append((inputs,e['text_mask'].cuda(),e['record']))
        if teacher_weight and not native_teacher_mode:
            from evals.context_retention_teacher import scope as teacher_scope
            labels,mask,checked=teacher_scope(e)
            if not torch.equal(mask,e['text_mask']):raise RuntimeError('Teacher retention changed existing mask')
            retention_targets.append(labels.cuda());retention_teacher_audits.append(checked)
    from evals.context_retention_teacher import policy as teacher_policy,loss as teacher_loss
    if native_teacher_mode:
        from evals.context_native_teacher_objective import prepare as prepare_native_teacher
        teacher_examples = examples
        if getattr(args, 'actual_native_teacher_cache', None):
            from evals.context_native_v2_teacher_objective import validate_configuration
            if not validate_configuration(args):
                raise ValueError('Repaired old25 teacher cache requires the bounded approved native-v2 recipe')
            from contextspan.model.sequence_convention import persona_prefix
            from evals.lychee_context import BASE_INSTRUCTIONS
            from evals.context_training_data import load_training
            teacher_examples, _ = load_training([args.actual_native_teacher_cache], spm,
                persona_prefix(BASE_INSTRUCTIONS, voice, spm), text_padding='legacy-epad0',
                request_policy='native-prelead-epad')
        teacher,native_teacher_batches=prepare_native_teacher(teacher_examples,spm,voice,retention,args)
    else:
        teacher=teacher_policy(teacher_weight,retention_teacher_audits,args.actual_retention_teacher_audit,
            args.assets,args.out/'voice.pt')
    additional_policy = None; additional_batches = []
    if getattr(args, 'native_v2_cache', None):
        from evals.context_native_v2_teacher_objective import prepare as prepare_native_v2
        additional_policy, additional_batches = prepare_native_v2(args, spm, voice, retention, teacher)
    audio49_policy=None;audio49_batches=[]
    if getattr(args,'native_audio49_audit',None):
        from evals.context_native_audio49_training import prepare as prepare_audio49
        audio49_policy,audio49_batches=prepare_audio49(args,spm,voice)
    stream_bound=None
    if getattr(args,'native_audio49_stream_auxiliary',False):
        from evals.context_native_stream_training import bindings as stream_bindings
        stream_bound=stream_bindings(args,native_teacher_batches,additional_batches)
    control_policy=None;control_bound=None
    if getattr(args,'control40_cache',None):
        from evals.native_control_stage import prepare as prepare_control40
        control_policy,control_bound=prepare_control40(args,warm_start)
    reasoning_policy=None;reasoning_bound=None
    if getattr(args,'reasoning32_cache',None):
        from evals.native_reasoning_stage import prepare as prepare_reasoning32
        reasoning_policy,reasoning_bound=prepare_reasoning32(args,warm_start,control_policy,audio49_policy)
    def forward(inputs): return lm.forward_codes(inputs)
    def digests():
        result = {}
        with torch.no_grad():
            for index,(audit,(inputs,mask,_)) in enumerate(zip(audits,batches)):
                z = forward(inputs)[1][0,0,mask].detach()
                if not torch.isfinite(z).all(): raise RuntimeError('Nonfinite valid temporal logits')
                result[str(index)+':'+audit['scenario']] = hashlib.sha256(z.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        return result
    def metrics():
        results = []
        with torch.no_grad():
            for index,(audit,(inputs,mask,target)) in enumerate(zip(audits,batches)):
                z = forward(inputs)[1][0,0]; pos = int(torch.where(mask)[0][0])
                y = target[mask]; z = z[mask].float()
                weights = torch.where(y==3,.5,1.)
                ce = F.cross_entropy(z,y,reduction='none')
                optimization_weights=text_weights(y,args.actual_response_first_word_weight,response_boundaries[index])
                content=(y>=5).nonzero().flatten()
                decision={}
                if len(content):
                    c=int(content[0]);probs=z[c].softmax(-1)
                    decision={'first_content_target':int(y[c]),
                        'first_content_top1':int(z[c].argmax()),
                        'first_content_probability':float(probs[y[c]]),
                        'first_content_request_probability':float(probs[4])}
                results.append({'scenario':audit['scenario'],'nll':float((ce*weights).sum()/weights.sum()),
                    **({'optimization_nll':float((ce*optimization_weights).sum()/optimization_weights.sum())} if decision else {}),
                    **decision,
                    'top1':float((z.argmax(-1)==y).float().mean()),
                    'non_pad_nll':float(ce[y!=3].mean()) if (y!=3).any() else None,
                    'silence_targets':int((y==3).sum()),
                    'silence_top1':float((z[y==3].argmax(-1)==3).float().mean()) if (y==3).any() else None,
                    'first_supervised_token_target':int(target[pos]),
                    'first_supervised_token_probability':float(z[0].softmax(-1)[target[pos]])})
        return results
    warm_policy=None;warm_references=[];clock_policy=None;clock_references=[]
    def state_metadata(frozen):
        return {'source_commit':SOURCE_COMMIT,
            'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            **({'source_transition':source_transition} if source_transition else {}),
            'assets_manifest_sha256':sha256(args.assets),'data_sha256':sha256(args.out/'data.json'),
            'native_protocol':{'name':'bootstrap-preserve-provided-v1',
                'code_sha256':sha256(Path(__file__).with_name('context_native_layout.py')),
                'attention_backend':'math'},
            'voice_sha256':sha256(args.out/'voice.pt'),'frozen_hashes':frozen,
            'retention_examples':[r[2] for r in retention],
            **({'warm_start':warm_start,'warm_start_optimizer_reset':not getattr(args,'native_audio49_parent_adam',False)} if warm_start else {}),
            'recipe':{'rank':8,'scaling':2.,'last_layers':4,'lr':rate,'weight_decay':0.,
                'gradient_clip':1.,'retention_kl_weight':base_retention_weight,'retention_hidden_mse_weight':10.*base_retention_weight,
                **({'retention_teacher_text':teacher} if teacher else {}),
                **({'additional_native_v2':additional_policy} if additional_policy else {}),
                **({'control40':control_policy} if control_policy else {}),
                **({'reasoning32':reasoning_policy} if reasoning_policy else {}),
                **({'warm_ordinary_retention':warm_policy} if warm_policy else {}),
                **({'warm_ordinary_clock_retention':clock_policy} if clock_policy else {}),
                'text_pad3_loss_weight':.5,
                **({'response_first_word_loss':decision} if decision else {}),
                **({'retention_sampling':{'policy':'rotate all retention examples within each verified kind',
                    'code_sha256':sha256(Path(__file__).with_name('context_component_kind.py'))}} if balance_retention else {}),
                'data_order':('round-robin verified kinds, then examples per kind; retention rotates within each kind'
                    if balance_retention else 'round-robin verified kinds, then examples per kind; retention step modulo four'
                    if args.actual_sampling=='balanced-kind' else 'actual step modulo component count; retention step modulo four'),
                **({'component_kind_order':component_kinds} if args.actual_sampling=='balanced-kind' else {}),
                **({'assistant_audio_auxiliary':audio49_policy,'original_actual_history_audio_loss_targets':0}
                    if audio49_policy else {}),
                'actual_component_count':len(batches),'past_loss_targets':0,
                'audio_loss_targets':audio49_policy['assistant_audio_loss_targets_per8'] if audio49_policy else 0,
                'execution':'BF16 batch MATH, no compilation or CUDA graphs'}}
    with no_compile(), no_cuda_graph(), sdpa_kernel(SDPBackend.MATH):
        baseline = digests(); baseline_metrics = metrics()
        frozen = tensor_hashes(lm.named_parameters())
        references = []
        with torch.no_grad():
            for inputs,mask,_ in retention:
                h,z = forward(inputs)
                references.append((h[:,mask].detach().clone(),z[0,0,mask].float().softmax(-1).detach()))
        rng = torch.cuda.get_rng_state().clone()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(17); torch.cuda.manual_seed_all(17)
            modules = install_adapters(lm,last_layers=4)
        if not torch.equal(rng,torch.cuda.get_rng_state()) or digests() != baseline:
            raise RuntimeError('Zero temporal adapters changed native logits or RNG')
        learned = adapter_parameters(lm)
        if {n for n,p in lm.named_parameters() if p.requires_grad} != set(learned):
            raise RuntimeError('Unexpected temporal trainable parameters')
        if control_policy:
            from evals.context_execution import snapshot as execution_snapshot
            actual_execution=execution_snapshot(lm)
            if any(actual_execution[k]!=v for k,v in control_policy['execution_required'].items()):
                raise RuntimeError('Control40 fit must use the independently checked native execution policy')
        if getattr(args,'native_v2_warm_ordinary_weight',0.):
            from evals.context_warm_ordinary_retention import prepare as warm_prepare
            warm_policy,warm_references=warm_prepare(lm,learned,native_teacher_batches,additional_batches,
                args,forward,state_metadata(frozen))
        if getattr(args,'native_v2_warm_ordinary_clock',False):
            from evals.context_warm_ordinary_clock import prepare as clock_prepare
            clock_policy,clock_references=clock_prepare(lm,learned,native_teacher_batches,additional_batches,
                batches,audits,args,forward,warm_policy)
        if args.reload:
            report = json.loads((args.out/'report.json').read_text())
            if reasoning_policy:
                from evals.native_reasoning_stage import SCHEMA as expected_schema
            elif control_policy:
                from evals.native_control_stage import SCHEMA as expected_schema
            else:expected_schema='dl-context-actual-audio49-adapter-proof-v1' if audio49_policy else 'dl-context-actual-adapter-proof-v1'
            if (report['schema'] != expected_schema or report['frozen_hashes'] != frozen
                    or report['retention_examples'] != [r[2] for r in retention]
                    or report['data_sha256'] != sha256(args.out/'data.json')
                    or report['voice_sha256'] != sha256(args.out/'voice.pt')
                    or report['assets_manifest_sha256'] != sha256(args.assets)
                    or report['actual_temporal_audit_sha256'] != sha256(pre/'actual-temporal-audit.json')
                    or report['checkpoint_sha256'] != sha256(args.out/'adapters.safetensors')):
                raise RuntimeError('Changed actual adapter proof')
            state = load_file(args.out/'adapters.safetensors',device='cuda')
            if warm_policy and (report.get('warm_ordinary_retention')!=warm_policy
                    or report.get('warm_ordinary_reference_audit_sha256')!=sha256(args.out/'warm-ordinary-reference-audit.json')):
                raise RuntimeError('Warm ordinary anchor changed before independent full-state restore')
            if clock_policy and (report.get('warm_ordinary_clock_retention')!=clock_policy
                    or report.get('warm_ordinary_clock_reference_audit_sha256')!=sha256(args.out/'warm-ordinary-clock-reference-audit.json')):
                raise RuntimeError('Ordinary clock/native-audio anchor changed before full-state restore')
            if set(state) != set(learned): raise RuntimeError('Incomplete actual adapter keys')
            with torch.no_grad():
                for n,p in learned.items():p.copy_(state[n])
            resume_verified = None
            if report.get('training_state_sha256'):
                if report['training_state_sha256'] != sha256(args.out/'training-state.pt'):
                    raise RuntimeError('Changed portable training state')
                before = tensor_hashes(learned.items())
                opt = torch.optim.AdamW(list(learned.values()),lr=rate,weight_decay=0.)
                snapshot = restore_state(args.out/'training-state.pt',learned,opt,state_metadata(frozen))
                resume_verified = (snapshot['completed_steps']==report['optimizer_steps']
                    and before==tensor_hashes(learned.items()))
                if not resume_verified:raise RuntimeError('Portable state differs from saved adapter')
            if audio49_policy:
                if report.get('assistant_audio_auxiliary')!=audio49_policy:raise RuntimeError('Audio49 policy reconstruction changed')
                from training.control.train_context import assert_equal,cpu_tree
                import random,numpy as np
                assert_equal(snapshot['adapters'],load_file(args.out/'adapters.safetensors',device='cpu'))
                assert_equal(snapshot['optimizer'],cpu_tree(opt.state_dict()))
                assert_equal(snapshot['torch_cpu_rng'],torch.get_rng_state())
                assert_equal(snapshot['torch_cuda_rng'],torch.cuda.get_rng_state_all())
                assert_equal(snapshot['python_rng'],random.getstate())
                n=np.random.get_state();assert_equal(snapshot['numpy_rng'],dict(name=n[0],keys=n[1].tolist(),position=int(n[2]),has_gauss=int(n[3]),cached_gauss=float(n[4])))
                from evals.context_native_audio49_training import fingerprints
                audio49_actual=fingerprints(lm,audio49_batches)
                if audio49_actual!=report['expected_audio49_logits']:raise RuntimeError('All49 acoustic restoration logits changed')
            stream_actual=None
            if stream_bound is not None:
                from evals.context_native_stream_training import fingerprints as stream_fingerprints
                stream_actual=stream_fingerprints(lm,stream_bound)
                if stream_actual!=report['expected_native_stream49_logits']:raise RuntimeError('Independent all49 native stream restore changed')
            control_actual=None
            if control_policy:
                from evals.native_control_training import fingerprints as control_fingerprints
                control_actual=control_fingerprints(lm,control_bound)
                if control_actual!=report['expected_native_control40_logits']:
                    raise RuntimeError('Independent all40 control text/audio restoration changed')
            reasoning_actual=None
            if reasoning_policy:
                from evals.native_reasoning_training import fingerprints as reasoning_fingerprints
                reasoning_actual=reasoning_fingerprints(lm,reasoning_bound)
                if reasoning_actual!=report['expected_native_reasoning32_logits']:
                    raise RuntimeError('Independent all32 reasoning text/audio restoration changed')
            actual = digests(); passed = actual == report['expected_logits']
            result = {'schema':('dl-context-actual-audio49-adapter-reload-v1' if audio49_policy else 'dl-context-actual-adapter-reload-v1'),'fresh_process':True,
                'all_valid_logits_exact':passed,'actual':actual,'checkpoint_sha256':report['checkpoint_sha256'],
                'data_sha256':report['data_sha256'],'optimizer_rng_state_restored':resume_verified,
                'training_state_sha256':report.get('training_state_sha256'),'passed':passed}
            if audio49_policy:
                result.update(pid=__import__('os').getpid(),training_pid=report['training_pid'],
                    all49_auxiliary_logits_exact=audio49_actual==report['expected_audio49_logits'],actual_audio49_logits=audio49_actual,
                    optimizer_and_rng_independently_compared_exact=True)
                if result['pid']==result['training_pid']:raise RuntimeError('Audio49 restore must use a fresh process')
            if stream_actual is not None:
                result.update(all49_native_stream_logits_exact=True,actual_native_stream49_logits=stream_actual)
            if control_actual is not None:
                from evals.native_control_stage import RELOAD_SCHEMA
                result.update(schema=RELOAD_SCHEMA,all40_native_control_logits_exact=True,
                              actual_native_control40_logits=control_actual)
            if reasoning_actual is not None:
                from evals.native_reasoning_stage import RELOAD_SCHEMA
                result.update(schema=RELOAD_SCHEMA,all32_native_reasoning_logits_exact=True,
                              actual_native_reasoning32_logits=reasoning_actual)
            if getattr(args,'native_audio49_parent_adam',False):
                clocks=sorted(set(int(x['step']) for x in opt.state.values()))
                if clocks!=[528] or snapshot['completed_steps']!=8:raise RuntimeError('Independent restore lost stage8/Adam528 clocks')
                result['parent_adam_parameter_steps_restored']=clocks
            (args.out/'reload-report.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
            if not passed: raise RuntimeError('Fresh actual adapter logits changed')
            return
        optimizer = torch.optim.AdamW(list(learned.values()),lr=rate,weight_decay=0.)
        initial_step = 0
        if warm_start and not args.resume_training:
            if parent['frozen_hashes']!=frozen:raise RuntimeError('Warm-start frozen base differs')
            state=load_file(args.warm_start_proof/'adapters.safetensors',device='cuda')
            if set(state)!=set(learned):raise RuntimeError('Warm-start adapter parameter coverage changed')
            with torch.no_grad():
                for n,p in learned.items():
                    if state[n].shape!=p.shape or state[n].dtype!=p.dtype or not torch.isfinite(state[n]).all():
                        raise RuntimeError('Invalid warm-start adapter tensor')
                    p.copy_(state[n])
            if optimizer.state:raise RuntimeError('New corrective stage unexpectedly retained optimizer moments')
        parent_adam=None
        if getattr(args,'native_audio49_parent_adam',False):
            from evals.context_native_parent_adam import port
            parent_adam=port(args.warm_start_proof,learned,optimizer,rate,args.out)
            if not args.native_audio49_preflight_only and sha256(args.out/'parent-adam-import.json')!=sha256(args.native_audio49_preflight/'parent-adam-import.json'):raise RuntimeError('Independent training/import moments differ from readiness')
        if args.resume_training:
            if args.actual_fit_updates>8:
                parent = json.loads((args.resume_training.parent/'report.json').read_text())
                loaded = json.loads((args.resume_training.parent/'reload-report.json').read_text())
                if (not parent['passed'] or not loaded['passed'] or not loaded['optimizer_rng_state_restored']
                        or parent['training_state_sha256'] != sha256(args.resume_training)
                        or loaded['training_state_sha256'] != parent['training_state_sha256']):
                    raise RuntimeError('Longer resumed adaptation lacks a verified portable preflight')
            snapshot = restore_state(args.resume_training,learned,optimizer,state_metadata(frozen),
                source_transition=source_transition)
            initial_step = snapshot['completed_steps']
            if source_transition and initial_step+args.actual_fit_updates!=40:
                raise RuntimeError('Owner exploration must stop after forty total optimizer updates')
        if getattr(args, 'native_v2_gradient_audit', False):
            from evals.context_native_gradient_audit import run as gradient_audit
            gradient_audit(lm, learned, batches, retention, references, native_teacher_batches,
                additional_batches, component_kinds, response_boundaries, audits, args,
                state_metadata(frozen))
            return
        if getattr(args, 'native_v2_retention_drift_proof', None):
            from evals.context_native_retention_drift import run as retention_drift
            if optimizer.state or initial_step:
                raise RuntimeError('Retention drift cannot resume or execute optimizer updates')
            retention_drift(lm, learned, native_teacher_batches, additional_batches,
                retention, references, args, state_metadata(frozen), forward)
            return
        if getattr(args,'native_v2_learning_curve_audit',False):
            from evals.context_native_learning_curve import run as learning_curve
            if optimizer.state or initial_step:raise RuntimeError('Learning curve cannot execute optimizer updates')
            learning_curve(lm,learned,batches,native_teacher_batches,additional_batches,retention,references,
                audits,args,state_metadata(frozen),forward)
            return
        if audio49_policy and args.native_audio49_preflight_only:
            if (optimizer.state and not parent_adam) or initial_step:raise RuntimeError('BF16 readiness cannot optimize or resume')
            if getattr(args,'native_audio49_stream_gradient_readiness',False):
                from evals.context_native_audio49_training import bf16_preflight
                from evals.context_native_complete_batch import preflight as complete_preflight
                from evals.context_native_stream_gradient import run as stream_readiness
                bf16_preflight(lm,learned,audio49_batches,audio49_policy,args)
                control_base={} if control_policy else None
                complete_digest=complete_preflight(lm,learned,args,audio49_policy,batches,retention,references,native_teacher_batches,additional_batches,component_kinds,response_boundaries,audits,warm_references,clock_references,audio49_batches,forward,args.actual_response_first_word_weight,capture_gradients=control_base)
                stream_readiness(lm,learned,native_teacher_batches,additional_batches,args,audio49_policy)
                control_ready=None;reasoning_ready=None
                if reasoning_policy:
                    from evals.native_reasoning_stage import combined_readiness
                    control_ready,reasoning_ready=combined_readiness(lm,learned,control_base,stream_bound,
                        audio49_policy,control_policy,control_bound,reasoning_policy,reasoning_bound,args)
                    del control_base
                elif control_policy:
                    from evals.native_control_stage import combined_readiness
                    control_ready=combined_readiness(lm,learned,control_base,stream_bound,audio49_policy,control_policy,control_bound,args)
                    del control_base
                path=args.out/'audio49-bf16-preflight.json';checked=json.loads(path.read_text())
                checked.update(complete_balanced_preflight_sha256=complete_digest,
                    native_stream_readiness_sha256=sha256(args.out/'native-stream49-gradient-readiness.json'),
                    peak_allocated_bytes=max(checked['peak_allocated_bytes'],json.loads((args.out/'complete-balanced-preflight.json').read_text())['peak_allocated_bytes'],json.loads((args.out/'native-stream49-gradient-readiness.json').read_text())['peak_allocated_bytes']))
                if control_ready:
                    checked['control40_combined_readiness_sha256']=control_ready
                    checked['peak_allocated_bytes']=max(checked['peak_allocated_bytes'],json.loads((args.out/'control40-combined-readiness.json').read_text())['peak_allocated_bytes'])
                if reasoning_ready:
                    checked['reasoning32_combined_readiness_sha256']=reasoning_ready
                    checked['peak_allocated_bytes']=max(checked['peak_allocated_bytes'],json.loads((args.out/'reasoning32-combined-readiness.json').read_text())['peak_allocated_bytes'])
                path.write_text(json.dumps(checked,indent=2)+'\n')
                return
            if getattr(args,'native_audio49_bf16_stream_diagnostic',False):
                from evals.context_native_bf16_stream_diagnostic import run as stream_diagnostic
                stream_diagnostic(lm,native_teacher_batches,additional_batches,args)
                return
            if getattr(args,'native_audio49_gradient_probe',False):
                from evals.context_native_audio49_gradient_probe import run as gradient_probe
                gradient_probe(lm,learned,batches,retention,references,native_teacher_batches,additional_batches,component_kinds,response_boundaries,audits,
                    warm_references,clock_references,audio49_batches,audio49_policy,args,state_metadata(frozen),forward)
                return
            from evals.context_native_audio49_training import bf16_preflight
            bf16_preflight(lm,learned,audio49_batches,audio49_policy,args)
            if getattr(args,'native_audio49_complete_batch',False):
                from evals.context_native_complete_batch import preflight as complete_preflight
                digest=complete_preflight(lm,learned,args,audio49_policy,batches,retention,references,native_teacher_batches,additional_batches,component_kinds,response_boundaries,audits,warm_references,clock_references,audio49_batches,forward,args.actual_response_first_word_weight)
                path=args.out/'audio49-bf16-preflight.json';checked=json.loads(path.read_text());checked['complete_balanced_preflight_sha256']=digest;checked['peak_allocated_bytes']=json.loads((args.out/'complete-balanced-preflight.json').read_text())['peak_allocated_bytes'];path.write_text(json.dumps(checked,indent=2)+'\n')
            return
        initial_metrics=metrics()
        steps = []; seen_a = seen_b = False; start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for local_step in range(args.actual_fit_updates):
            warm_rows=[];clock_rows=[];audio49_rows=[]
            additional_loss = None; additional_record = None; additional_records = []
            step = initial_step+local_step
            optimizer.zero_grad(set_to_none=True)
            from evals.context_component_kind import balanced_index
            index=balanced_index(component_kinds,step) if args.actual_sampling=='balanced-kind' else step%len(batches)
            inputs,mask,target = batches[index]
            if getattr(args,'native_audio49_complete_batch',False):
                from evals.context_native_complete_batch import backward as complete_backward
                from evals.context_component_kind import retention_index
                complete_groups=complete_backward(lm,batches,retention,references,native_teacher_batches,additional_batches,component_kinds,response_boundaries,audits,warm_references,clock_references,audio49_batches,forward,args.actual_response_first_word_weight)
                def mean(group,field='loss'):return torch.tensor(sum(x['weight']*x[field] for x in complete_groups[group]))
                ce=mean('actual');kl=mean('reference','kl');hidden=mean('reference','hidden_mse');supervised=mean('old25')
                ri=retention_index(component_kinds,step,len(retention),balance_retention)
                ti=-1;trecord=dict(id='all25-balanced-kind-mean',kind='all4');teacher_breakdown=None
                teacher_target_count=sum(x['targets'] for x in complete_groups['old25'])
                additional_records=complete_groups['new24'];audio49_rows=complete_groups['audio49']
                warm_rows=complete_groups['word'];clock_rows=complete_groups['clock']
            else:
                with checkpoint_language_layers(lm):
                    logits = forward(inputs)[1][0,0,mask].float()
                    y = target[mask]; weights = text_weights(y,args.actual_response_first_word_weight,response_boundaries[index])
                    ce = (F.cross_entropy(logits,y,reduction='none')*weights).sum()/weights.sum()
                    ce.backward()
                    from evals.context_component_kind import retention_index
                    ri=retention_index(component_kinds,step,len(retention),balance_retention)
                    inputs,mask,_ = retention[ri]
                    h,z = forward(inputs); base_h,prob = references[ri]
                    kl = F.kl_div(z[0,0,mask].float().log_softmax(-1),prob,reduction='batchmean')
                    hidden = (h[:,mask].float()-base_h.float()).square().mean()
                    supervised=teacher_loss(z[0,0],retention_targets[ri],mask) if teacher and not native_teacher_mode else None
                    teacher_target_count=int(mask.sum())
                    regularizer=kl+10.*hidden
                    if base_retention_weight!=1.:regularizer=base_retention_weight*regularizer
                    if teacher and not native_teacher_mode:regularizer=regularizer+teacher_weight*supervised
                    regularizer.backward()
                    if native_teacher_mode:
                        from evals.context_native_teacher_objective import teacher_index
                        ti=teacher_index(component_kinds,step,ri,[b[3]['kind'] for b in native_teacher_batches])
                        tinputs,tmask,ttarget,trecord=native_teacher_batches[ti]
                        tz=forward(tinputs)[1][0,0]
                        teacher_breakdown=None
                        if args.actual_native_teacher_loss=='balanced-spoken':
                            from evals.context_teacher_text_balance import loss as balanced_teacher_loss
                            supervised,teacher_breakdown=balanced_teacher_loss(tz,ttarget,tmask)
                        else:supervised=teacher_loss(tz,ttarget,tmask)
                        teacher_target_count=int(tmask.sum())
                        (teacher_weight*supervised).backward()
                    if additional_batches:
                        from evals.context_native_teacher_objective import teacher_index
                        from evals.context_teacher_text_balance import loss as balanced_teacher_loss
                        teacher_kinds=[b[3]['kind'] for b in additional_batches]
                        if getattr(args, 'native_v2_full24_per8', False):
                            from evals.context_native_teacher_coverage import complete_indices
                            indices=complete_indices(component_kinds,step,ri,teacher_kinds)
                        else:indices=[teacher_index(component_kinds,step,ri,teacher_kinds)]
                        additional_losses=[]
                        for ai in indices:
                            ainputs,amask,atarget,additional_record=additional_batches[ai]
                            az=forward(ainputs)[1][0,0]
                            loss,_=balanced_teacher_loss(az,atarget,amask)
                            if not torch.isfinite(loss):raise RuntimeError('Nonfinite additional native-v2 supervision')
                            (loss/len(indices)).backward()
                            additional_losses.append(loss.detach())
                            additional_records.append(dict(id=additional_record['id'],kind=additional_record['kind'],
                                text_ce=float(loss.detach()),weight=1/len(indices)))
                        additional_loss=torch.stack(additional_losses).mean()
                    if warm_references:
                        from evals.context_warm_ordinary_retention import loss as warm_loss
                        for winputs,wmask,reference,identity in warm_references:
                            wh,wz=forward(winputs);value,wkl,wmse=warm_loss(wh,wz,wmask,reference)
                            if not torch.isfinite(value):raise RuntimeError('Nonfinite additional ordinary retention')
                            (warm_policy['weight']/len(warm_references)*value).backward()
                            warm_rows.append(dict(id=identity,kl=float(wkl.detach()),hidden_mse=float(wmse.detach()),
                                weight=warm_policy['weight']/len(warm_references)))
                    if clock_references:
                        from evals.context_warm_ordinary_retention import loss as clock_loss
                        for cinputs,cmask,reference,identity in clock_references:
                            ch,cz=forward(cinputs);value,ckl,cmse=clock_loss(ch,cz,cmask,reference)
                            if not torch.isfinite(value):raise RuntimeError('Nonfinite ordinary clock/native-audio retention')
                            (clock_policy['weight']/len(clock_references)*value).backward()
                            clock_rows.append(dict(id=identity,kl=float(ckl.detach()),hidden_mse=float(cmse.detach()),
                                weight=clock_policy['weight']/len(clock_references)))
                    if audio49_policy:
                        from evals.context_native_audio49_training import backward as audio49_backward
                        audio49_rows=audio49_backward(lm,audio49_batches,local_step)
            native_stream_rows=[]
            if stream_bound is not None:
                from evals.context_native_stream_training import backward as stream_backward
                native_stream_rows=stream_backward(lm,stream_bound,audio49_policy['native_stream_auxiliary'],local_step)
            control_rows=[]
            if control_policy:
                from evals.native_control_training import backward as control_backward
                control_rows=control_backward(lm,control_bound,local_step)
            reasoning_rows=[]
            if reasoning_policy:
                from evals.native_reasoning_training import backward as reasoning_backward
                reasoning_rows=reasoning_backward(lm,reasoning_bound,local_step)
            if not all(torch.isfinite(v) for v in ((ce,kl,hidden,supervised) if teacher else (ce,kl,hidden))):
                raise RuntimeError('Nonfinite actual or retention loss')
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in learned.values()):
                raise RuntimeError('Missing/nonfinite actual temporal gradient')
            a = sum(int(torch.count_nonzero(p.grad)) for n,p in learned.items() if '.lora_A.' in n)
            b = sum(int(torch.count_nonzero(p.grad)) for n,p in learned.items() if '.lora_B.' in n)
            seen_a |= a>0; seen_b |= b>0
            norm = float(torch.nn.utils.clip_grad_norm_(list(learned.values()),1.))
            if not math.isfinite(norm): raise RuntimeError('Nonfinite actual gradient norm')
            optimizer.step()
            save_state(args.out/'training-state.pt',learned,optimizer,step+1,state_metadata(frozen))
            row = {'step':step+1,'scenario':audits[index]['scenario'],'component_index':index,
                'supervision_kind':component_kinds[index],
                'future_text_ce':float(ce.detach()),'retention_kind':retention[ri][2]['kind'],
                **({'retention_index':ri} if balance_retention else {}),
                'retention_kl':float(kl.detach()),'retention_hidden_mse':float(hidden.detach()),
                **({'retention_teacher_text_ce':float(supervised.detach()),
                    'retention_teacher_text_targets':teacher_target_count} if teacher else {}),
                **({'native_teacher_index':ti,'native_teacher_id':trecord['id'],
                    'native_teacher_kind':trecord['kind']} if native_teacher_mode else {}),
                **({'native_teacher_loss_breakdown':teacher_breakdown}
                    if native_teacher_mode and teacher_breakdown is not None else {}),
                **({'additional_native_v2_id':additional_record['id'],
                    'additional_native_v2_kind':additional_record['kind'],
                    'additional_native_v2_text_ce':float(additional_loss.detach())}
                    if additional_record and not getattr(args, 'native_v2_full24_per8', False) else {}),
                **({'additional_native_v2_microbatch':additional_records}
                    if getattr(args, 'native_v2_full24_per8', False) else {}),
                **({'warm_ordinary_microbatch':warm_rows} if warm_rows else {}),
                **({'warm_ordinary_clock_microbatch':clock_rows} if clock_rows else {}),
                **({'assistant_audio49_microbatch':audio49_rows} if audio49_policy else {}),
                'nonzero_A_gradients':a,'nonzero_B_gradients':b,'gradient_norm_before_clip':norm}
            if getattr(args,'native_audio49_complete_batch',False):
                row.update(scenario='all28-balanced-kind-mean',component_index=None,supervision_kind='all8-balanced-kinds',retention_kind='all4-reference-mean',retention_index=None,native_teacher_index=None,complete_balanced_microbatches=complete_groups,legacy_nominal_schedule_anchor=dict(component_index=index,retention_index=ri))
            if stream_bound is not None:row['native_stream_teacher_microbatch']=native_stream_rows
            if control_policy:row['control40_microbatch']=control_rows
            if reasoning_policy:row['reasoning32_microbatch']=reasoning_rows
            steps.append(row); print(json.dumps(row),flush=True)
        if getattr(args,'native_audio49_complete_batch',False):
            from evals.context_native_complete_batch import validate_exposure as validate_complete
            validate_complete(steps,audio49_policy['teacher_ids'],audio49_policy['source_unique49_acoustic_targets'],audio49_policy['complete_batch_membership'])
        else:
            if getattr(args, 'native_v2_full24_per8', False):
                expected_ids={b[3]['id'] for b in additional_batches}
                for offset in range(0,len(steps),8):
                    exposed=[view['id'] for row in steps[offset:offset+8] for view in row['additional_native_v2_microbatch']]
                    if len(exposed)!=24 or set(exposed)!=expected_ids:
                        raise RuntimeError('Full24 native pilot omitted or duplicated a teacher in an eight-update block')
            if audio49_policy:
                exposed=[row['id'] for step in steps for row in step['assistant_audio49_microbatch']]
                if len(exposed)!=49 or set(exposed)!=set(audio49_policy['teacher_ids']):raise RuntimeError('Incomplete49 assistant acoustic exposure')
                count=sum(sum(row['objective']['assistant_audio_targets']) for step in steps for row in step['assistant_audio49_microbatch'])
                if count!=audio49_policy['assistant_audio_loss_targets_per8']:raise RuntimeError('Audio49 target validity changed from independent numerical audit')
        if stream_bound is not None:
            from evals.context_native_stream_training import exposure as stream_exposure
            stream_exposure(steps,audio49_policy['native_stream_auxiliary'])
        if control_policy:
            from evals.native_control_training import exposure as control_exposure
            control_exposure(steps,control_policy)
        if reasoning_policy:
            from evals.native_reasoning_training import exposure as reasoning_exposure
            reasoning_exposure(steps,reasoning_policy)
        if not seen_a or not seen_b: raise RuntimeError('Dead actual temporal adapter')
        if frozen != tensor_hashes((n,p) for n,p in lm.named_parameters() if not p.requires_grad):
            raise RuntimeError('Frozen speech/base weights changed')
        expected = digests()
        if expected == baseline: raise RuntimeError('Actual temporal updates changed no valid logits')
        save_file({n:p.detach().cpu().contiguous() for n,p in learned.items()},args.out/'adapters.safetensors')
        report = {'schema':'dl-context-actual-adapter-proof-v1','source_commit':SOURCE_COMMIT,
            **({'source_transition':source_transition,'owner_exploration_training_only':True} if source_transition else {}),
            'repo_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'assets_manifest_sha256':sha256(args.assets),'data_sha256':sha256(args.out/'data.json'),
            'voice_sha256':sha256(args.out/'voice.pt'),'voice_reference_sha256':sha256(args.voice_reference),
            'checkpoint_sha256':sha256(args.out/'adapters.safetensors'),
            'actual_temporal_audit_sha256':sha256(pre/'actual-temporal-audit.json'),
            'actual_temporal_preflight':str(pre),'all_actual_temporal_audits_passed':True,
            'adapter_modules':modules,'last_layers':4,'rank':8,'scaling':2.,
            'trainable_parameters':sum(p.numel() for p in learned.values()),'optimizer_steps':initial_step+len(steps),
            'updates_this_run':len(steps),'initial_optimizer_steps':initial_step,
            'resumed_from_training_state':str(args.resume_training) if args.resume_training else None,
            **({'warm_start':warm_start,'warm_start_optimizer_reset':not getattr(args,'native_audio49_parent_adam',False)} if warm_start else {}),
            'training_state_sha256':sha256(args.out/'training-state.pt'),
            'past_loss_targets':0,'audio_loss_targets':0,'all_component_audits_passed':True,
            'native_protocol':state_metadata(frozen)['native_protocol'],
            'zero_adapter_native_batch_logits_exact':True,'adapter_construction_rng_preserved':True,
            'frozen_weights_unchanged':True,'frozen_hashes':frozen,'base_audio_parameters_frozen':True,
            'retention_examples':[r[2] for r in retention],
            'retention_objective':('independently audited native TRAIN teacher text CE; base KL and hidden MSE weights zero'
                if not base_retention_weight else 'KL plus 10x hidden MSE and explicitly audited full TRAIN teacher text CE'
                if teacher else 'KL on unchanged base text distribution plus 10x hidden MSE, approved independent TRAIN histories'),
            **({'base_retention_weight':base_retention_weight} if base_retention_weight!=1. else {}),
            **({'retention_teacher_text':teacher} if teacher else {}),
            **({'additional_native_v2':additional_policy} if additional_policy else {}),
            **({'warm_ordinary_retention':warm_policy,
                'warm_ordinary_reference_audit_sha256':sha256(args.out/'warm-ordinary-reference-audit.json')} if warm_policy else {}),
            **({'warm_ordinary_clock_retention':clock_policy,
                'warm_ordinary_clock_reference_audit_sha256':sha256(args.out/'warm-ordinary-clock-reference-audit.json')} if clock_policy else {}),
            **({'response_first_word_loss':decision} if decision else {}),
            'baseline_metrics':baseline_metrics,'initial_stage_metrics':initial_metrics,
            'final_metrics':metrics(),'expected_logits':expected,'steps':steps,
            'execution':'native BF16 batch MATH attention, no compilation/graphs; actual FP32 causal preflight required',
            'training_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'quality_qualified':False,'free_running_result_use_proven':False,
            'audio_behavior_preservation_proven':False,'passed':True}
        if parent_adam:
            report.update(parent_adam_import_sha256=sha256(args.out/'parent-adam-import.json'),
                adam_parameter_steps_after=sorted(set(int(x['step']) for x in optimizer.state.values())),
                optimizer_steps_scope='Updates in this corrective stage; preserved Adam clocks are recorded separately')
            if report['adam_parameter_steps_after']!=[528]:raise RuntimeError('Parent Adam clock did not advance exactly eight updates')
        if warm_policy:
            if any([r['id'] for r in step.get('warm_ordinary_microbatch',[])]!=[r['id'] for r in warm_policy['examples']] for step in steps):
                raise RuntimeError('Every update must retain all ten approved ordinary anchors')
            report['retention_objective']+='; additional64x mean native520 ordinary-word KL plus10hiddenMSE, legacy losses unchanged'
        if clock_policy:
            if any([r['id'] for r in step.get('warm_ordinary_clock_microbatch',[])]!=clock_policy['ids'] for step in steps):
                raise RuntimeError('Every update must preserve all14 approved ordinary clock/native-audio anchors')
            report['retention_objective']+='; separate64x mean ordinary clock/native-audio KL plus10hiddenMSE, word/legacy losses unchanged'
        if audio49_policy:
            from evals.context_native_audio49_training import fingerprints,SCHEMA
            report.update(schema=SCHEMA,assistant_audio_auxiliary=audio49_policy,combined_recipe=state_metadata(frozen)['recipe'],
                original_actual_history_audio_loss_targets=0,user_audio_loss_targets=0,prefix_audio_loss_targets=0,supplied_span_audio_loss_targets=0,
                audio_loss_targets=sum(sum(row['objective']['assistant_audio_targets']) for step in steps for row in step['assistant_audio49_microbatch']),
                expected_audio49_logits=fingerprints(lm,audio49_batches),training_pid=__import__('os').getpid())
            report['retention_objective']+='; additional assistant-only acoustic CE on all49 approved teachers, every original objective retained'
        if stream_bound is not None:
            from evals.context_native_stream_training import fingerprints as stream_fingerprints
            report['expected_native_stream49_logits']=stream_fingerprints(lm,stream_bound)
            report['execution']='Native BF16 batch MATH original154 plus recorded-call streaming49 teacher CE; truncated call-boundary autograd; no compilation or CUDA graphs'
            report['training_seconds_scope']='Eight updates plus post-fit frozen-weight/digest checks, checkpoint writing and original28 final metrics; excludes final acoustic49/native49 fingerprints and fresh-process restoration'
            report['retention_objective']+='; supplemental balanced49 native-call teacher text CE with declared truncated autograd; original154 unchanged'
        if control_policy:
            from evals.native_control_stage import SCHEMA
            from evals.native_control_training import fingerprints as control_fingerprints
            control_targets=sum(sum(x['acoustic_objective']['assistant_audio_targets']) for step in steps for x in step['control40_microbatch'])
            text_targets=sum(x['text_targets'] for step in steps for x in step['control40_microbatch'])
            if control_targets!=control_policy['assistant_audio_loss_targets_per8'] or text_targets!=control_policy['text_loss_targets_per8']:
                raise RuntimeError('Control40 complete source target count changed')
            report.update(schema=SCHEMA,control40=control_policy,
                control40_audio_loss_targets=control_targets,control40_text_loss_targets=text_targets,
                total_assistant_audio_loss_targets=report['audio_loss_targets']+control_targets,
                audio_loss_targets_scope='Original49 assistant acoustic objective; additional control40 and total counts are explicit separate fields',
                expected_native_control40_logits=control_fingerprints(lm,control_bound),
                control40_combined_readiness_sha256=sha256(args.out/'approved-control40-combined-readiness.json'))
            report['execution']+='; supplemental all40 native-call text plus assistant-only acoustic CE'
            report['training_seconds_scope']+='; also excludes final all40 native text/acoustic fingerprints'
        if reasoning_policy:
            from evals.native_reasoning_stage import SCHEMA
            from evals.native_reasoning_training import fingerprints as reasoning_fingerprints
            reasoning_targets=sum(sum(x['acoustic_objective']['assistant_audio_targets']) for step in steps for x in step['reasoning32_microbatch'])
            reasoning_text=sum(x['text_targets'] for step in steps for x in step['reasoning32_microbatch'])
            if (reasoning_targets!=reasoning_policy['assistant_audio_loss_targets_per8']
                    or reasoning_text!=reasoning_policy['text_loss_targets_per8']):
                raise RuntimeError('Reasoning32 exact source target counts changed')
            report.update(schema=SCHEMA,reasoning32=reasoning_policy,
                reasoning32_audio_loss_targets=reasoning_targets,reasoning32_text_loss_targets=reasoning_text,
                all_training_assistant_audio_loss_targets=report['total_assistant_audio_loss_targets']+reasoning_targets,
                total_assistant_audio_loss_targets_scope='Retained original49 plus control40 subtotal; all_training_assistant_audio_loss_targets includes reasoning32',
                expected_native_reasoning32_logits=reasoning_fingerprints(lm,reasoning_bound),
                reasoning32_combined_readiness_sha256=sha256(args.out/'approved-reasoning32-combined-readiness.json'))
            report['execution']+='; supplemental all32 reasoning native-call text plus pinned-span-mask assistant acoustic CE'
            report['training_seconds_scope']+='; also excludes final all32 native text/acoustic fingerprints'
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('passed','checkpoint_sha256','final_metrics','peak_allocated_bytes')}),flush=True)
