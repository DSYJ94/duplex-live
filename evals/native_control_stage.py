"""Bind control40 to the retained154/native49 bounded trainer and fresh restore.

The original objectives and optimizer are untouched. Readiness reuses the SAME
validated original154 gradient at a fixed parameter point across eight distinct
supplemental partitions; no optimizer runs during that calculation.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from evals.moshi_model_download import sha256

SCHEMA='dl-native-control40-retained154-stream49-proof-v1'
RELOAD_SCHEMA='dl-native-control40-retained154-stream49-reload-v1'
OPTIONS=('control40_cache','control40_engine','control40_numerical','control40_gradient',
         'control40_parent_review','control40_development_evidence')


def validate(args):
    supplied=[bool(getattr(args,n,None)) for n in OPTIONS]
    if not any(supplied):return False
    if (not all(supplied) or not args.native_audio49_audit or not args.native_audio49_complete_batch
            or not args.native_audio49_stream_auxiliary or not args.native_audio49_parent_adam
            or args.actual_fit_updates!=8 or args.resume_training or args.native_audio49_step_scale!=.1
            or not args.warm_start_proof or Path(args.warm_start_proof).name!='context-training-proof-7510331'
            or (args.native_audio49_preflight_only and not args.native_audio49_stream_gradient_readiness)):
        raise ValueError('Control40 requires one new8 native520 stage with complete154/native49 and independent combined readiness')
    return True


def selection(args,parent):
    from evals.context_native_extension_review import committed_document
    chosen=committed_document(args.control40_parent_review)
    if (chosen['schema']!='dl-native-control40-bounded-repair-parent-selection-v1'
            or chosen['parent_proof']!=Path(args.warm_start_proof).name
            or chosen['parent_checkpoint_sha256']!=parent['checkpoint_sha256']
            or chosen['parent_training_state_sha256']!=parent['training_state_sha256']
            or chosen['parent_adam_clock']!=520 or chosen['complete_native_cases_reviewed']!=40
            or chosen['complete_independent_speech_cases_reviewed']!=40
            or chosen['baseline_is_production_qualified'] or chosen['candidate7590052_adopted']
            or chosen['optimizer_start_ready']):
        raise ValueError('Exact committed complete40 parent review required; never an adoption stamp')
    root=Path(__file__).resolve().parents[1]
    for key,native,speech in (
        ('default_review','native10-paired-v1','native10-independent-speech-v1'),
        ('archive_review','native10-paired-archive-tf32-v1','native10-independent-speech-archive-tf32-v1')):
        review=committed_document(root/chosen[key])
        if not review['complete20_native_cases_reviewed'] or not review['complete20_independent_speech_reviewed']:
            raise ValueError('Incomplete paired development review')
        for role in ('baseline520','candidate7590052'):
            model=review['models'][role];run=args.control40_development_evidence/native/role
            audit=args.control40_development_evidence/speech/role/'report.json'
            if sha256(audit)!=model['independent_speech_report_sha256']:
                raise ValueError('Changed independent complete speech10 evidence')
            measured=json.loads(audit.read_text())['results']
            if len(measured)!=10 or [c['index'] for c in model['cases']]!=list(range(10)):
                raise ValueError('Every development case must be retained')
            for index,(case,audio) in enumerate(zip(model['cases'],measured)):
                folder=run/('case-%02d'%index);report=json.loads((folder/'report.json').read_text())
                if (sha256(folder/'report.json')!=case['run_report_sha256']
                        or sha256(folder/'model.wav')!=case['audio_sha256']
                        or audio['wav_sha256']!=case['audio_sha256'] or report['text']!=case['sampled_text']
                        or report['scheduled_dispatch'] or report['word_or_acoustic_tokens_forced']
                        or report['response_start_forced'] or not case['reason'].strip()):
                    raise ValueError('Changed/forced/incomplete full native development evidence')
    return chosen


def prepare(args,parent):
    from evals.native_control_training import prepare as prepare_control
    policy,bound=prepare_control(args.control40_cache,args.control40_engine,args.control40_numerical,
                                 args.control40_gradient,parent)
    selection(args,parent)
    manifest=json.loads((args.control40_engine/'engine-inputs.json').read_text())
    counts={r['id']:r['ownership']['text_targets'] for r in manifest['cases']}
    policy.update(assistant_audio_loss_targets_per8=sum(8*t-7 for t in counts.values()),
                  text_loss_targets_per8=sum(counts.values()),
                  parent_selection_sha256=sha256(args.control40_parent_review),
                  target_counts={key:dict(text=value,audio=[value]+[value-1]*7) for key,value in counts.items()},
                  stage_implementation_sha256=sha256(Path(__file__)))
    policy['execution_required']=dict(matmul_tf32=False,cudnn_tf32=True,math_sdp=True,
                                      flash_sdp=False,mem_efficient_sdp=False,cudnn_sdp=False,
                                      float32_precision='highest')
    sources={'approved-control40-cache.json':args.control40_cache/'cache.json',
             'approved-control40-engine.json':args.control40_engine/'engine-inputs.json',
             'approved-control40-numerical.json':args.control40_numerical/'numerical.json',
             'approved-control40-gradient.json':args.control40_gradient/'gradient.json',
             'control40-parent-selection.json':args.control40_parent_review}
    for name,path in sources.items():
        target=args.out/name
        if args.reload:
            if sha256(target)!=sha256(path):raise ValueError('Changed packaged control40 approval')
        else:target.write_bytes(path.read_bytes())
    if not args.native_audio49_preflight_only:
        path=args.native_audio49_preflight/'control40-combined-readiness.json'
        check_readiness(path,policy)
        target=args.out/'approved-control40-combined-readiness.json'
        if args.reload:
            if sha256(target)!=sha256(path):raise ValueError('Changed combined readiness before restore')
        else:target.write_bytes(path.read_bytes())
    return policy,bound


def check_readiness(path,policy,source_commit=None):
    from evals.native_control_training import exposure
    r=json.loads(path.read_text())
    if (r['schema']!='dl-native-control40-combined-readiness-v1' or r['passed'] is not True
            or r['policy']!=policy or r['all_parameters_unchanged'] is not True or r['optimizer_updates']!=0
            or r['checkpoint_written'] is not False or r['source_commit']!=(source_commit or os.environ['DL_TASK_COMMIT'])
            or r['peak_allocated_bytes']>=80_000_000_000 or r['original154_components']!=154
            or len(r['base154_gradient_sha256'])!=26 or len(r['steps'])!=8
            or r['fixed_parent154_gradient_reused_exactly'] is not True
            or any(not math.isfinite(s['combined_gradient_norm']) or s['combined_gradient_norm']<=0
                   or s['frozen_parameter_gradients']!=0 for s in r['steps'])):
        raise ValueError('Independent combined154/native49/control40 gradient readiness required')
    exposure(r['steps'],policy)
    exact_target_counts(r['steps'],policy)
    return r


def exact_target_counts(steps,policy):
    if set(policy['target_counts'])!={t['id'] for t in policy['teachers']}:
        raise ValueError('Every control teacher needs explicit source target counts')
    for step in steps:
        for row in step['control40_microbatch']:
            counts=policy['target_counts'][row['id']]
            if (row['text_targets']!=counts['text'] or row['acoustic_objective']['assistant_audio_targets']!=counts['audio']
                    or counts['audio']!=[counts['text']]+[counts['text']-1]*7):
                raise ValueError('Changed exact all-frame control40 training target counts')
    if (sum(c['text'] for c in policy['target_counts'].values())!=policy['text_loss_targets_per8']
            or sum(sum(c['audio']) for c in policy['target_counts'].values())!=policy['assistant_audio_loss_targets_per8']):
        raise ValueError('Control40 total target count changed')


def combined_readiness(lm,learned,base_gradients,stream_bound,retained_policy,policy,bound,args):
    import torch
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_native_stream_training import backward as stream_backward,exposure as stream_exposure
    from evals.native_control_training import backward,exposure
    if set(base_gradients)!=set(learned) or len(base_gradients)!=26:
        raise ValueError('Capture the actual validated full154 gradient before clearing it')
    before=tensor_hashes(lm.named_parameters());steps=[]
    base_hashes={n:hashlib.sha256(g.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                 for n,g in base_gradients.items()}
    for index in range(8):
        for n,p in learned.items():p.grad=base_gradients[n].detach().clone()
        old=stream_backward(lm,stream_bound,retained_policy['native_stream_auxiliary'],index)
        new=backward(lm,bound,index)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in learned.values()):
            raise ValueError('Missing/nonfinite combined retained and control gradient')
        frozen=sum(p.grad is not None for n,p in lm.named_parameters() if n not in learned)
        if frozen:raise ValueError('Frozen parameters acquired a combined gradient')
        norm=sum(float(p.grad.detach().double().square().sum()) for p in learned.values())**.5
        steps.append(dict(step=index+1,native_stream_teacher_microbatch=old,control40_microbatch=new,
                          combined_gradient_norm=norm,frozen_parameter_gradients=frozen))
        print(json.dumps(dict(control40_combined_partition=index+1,combined_gradient_norm=norm)),flush=True)
        for p in learned.values():p.grad=None
    exposure(steps,policy);stream_exposure(steps,retained_policy['native_stream_auxiliary'])
    unchanged=before==tensor_hashes(lm.named_parameters())
    r=dict(schema='dl-native-control40-combined-readiness-v1',policy=policy,steps=steps,
           passed=unchanged,all_parameters_unchanged=unchanged,
           optimizer_updates=0,checkpoint_written=False,training_approved=False,original154_components=154,
           base154_gradient_sha256=base_hashes,fixed_parent154_gradient_reused_exactly=True,
           original154_gradient_source_sha256=sha256(args.out/'complete-balanced-preflight.json'),
           native49_readiness_sha256=sha256(args.out/'native-stream49-gradient-readiness.json'),
           source_commit=os.environ['DL_TASK_COMMIT'],peak_allocated_bytes=torch.cuda.max_memory_allocated())
    path=args.out/'control40-combined-readiness.json';path.write_text(json.dumps(r,indent=2)+'\n')
    check_readiness(path,policy)
    return sha256(path)


def verify_checkpoint(proof):
    from evals.context_native_audio49_training import verify_retained_checkpoint
    from evals.native_control_training import exposure
    identity,r=verify_retained_checkpoint(proof,SCHEMA,RELOAD_SCHEMA)
    proof=Path(proof);v=json.loads((proof/'reload-report.json').read_text());policy=r['control40']
    exposure(r['steps'],policy)
    if (r['combined_recipe']['control40']!=policy or r['control40_audio_loss_targets']!=policy['assistant_audio_loss_targets_per8']
            or r['total_assistant_audio_loss_targets']!=r['audio_loss_targets']+r['control40_audio_loss_targets']
            or r['control40_text_loss_targets']!=policy['text_loss_targets_per8']
            or not v['all40_native_control_logits_exact'] or v['actual_native_control40_logits']!=r['expected_native_control40_logits']
            or set(r['expected_native_control40_logits'])!={t['id'] for t in policy['teachers']}
            or sha256(proof/'approved-control40-combined-readiness.json')!=r['control40_combined_readiness_sha256']
            or sha256(proof/'approved-control40-cache.json')!=policy['cache_sha256']
            or sha256(proof/'approved-control40-engine.json')!=policy['engine_sha256']
            or sha256(proof/'approved-control40-numerical.json')!=policy['numerical_sha256']
            or sha256(proof/'approved-control40-gradient.json')!=policy['gradient_readiness_sha256']
            or sha256(proof/'control40-parent-selection.json')!=policy['parent_selection_sha256']):
        raise ValueError('Incomplete retained154/native49/control40 proof or independent restoration')
    check_readiness(proof/'approved-control40-combined-readiness.json',policy,r['repo_commit'])
    import subprocess
    root=Path(__file__).resolve().parents[1]
    for name,key in (('native_control_stage.py','stage_implementation_sha256'),('native_control_training.py','implementation_sha256')):
        expected=hashlib.sha256(subprocess.check_output(['git','show',r['repo_commit']+':evals/'+name],cwd=root)).hexdigest()
        if policy[key]!=expected:raise ValueError('Changed committed control40 implementation')
    ids=[x['id'] for x in policy['teachers']]
    for name in ('engine','numerical','gradient'):
        approved=json.loads((proof/('approved-control40-'+name+'.json')).read_text())
        if (not approved['passed'] or not approved['complete40'] or approved['optimizer_updates']!=0
                or len(approved['cases'])!=40 or set(x['id'] for x in approved['cases'])!=set(ids)
                or any(not x['passed'] for x in approved['cases'])):
            raise ValueError('Incomplete packaged complete40 approval: '+name)
    exact_target_counts(r['steps'],policy)
    return identity,r
