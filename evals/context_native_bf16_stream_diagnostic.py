"""Measure actual BF16 streamed/batched teacher differences; never approve a fit."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def call_partition(calls, frames):
    cursor=0
    for call in calls:
        start,n=call['start'],call['frames']
        if type(start) is not int or type(n) is not int or start!=cursor or n<1:
            raise ValueError('Native calls must partition the complete approved input')
        cursor+=n
    if cursor!=frames:raise ValueError('Native calls omitted approved frames')


def validate(args):
    if (not args.native_audio49_preflight_only or not args.native_audio49_audit
            or args.native_audio49_preflight or args.reload or args.resume_training
            or args.native_audio49_gradient_probe or args.native_audio49_parent_adam
            or args.native_audio49_complete_batch
            or Path(args.warm_start_proof).name!='context-training-proof-7510331'):
        raise ValueError('BF16 streaming diagnostic requires an independent zero-update native520 preflight')


def compare(hidden,logits,stream_hidden,stream_logits,mask,targets):
    import torch
    from torch.nn import functional as F
    if hidden.shape!=stream_hidden.shape or logits.shape!=stream_logits.shape:
        raise ValueError('Streamed and batched output shapes differ')
    if not all(torch.isfinite(t).all() for t in (hidden,logits,stream_hidden,stream_logits)):
        raise ValueError('Nonfinite BF16 diagnostic outputs')
    z=logits[0,0,mask].float();s=stream_logits[0,0,mask].float();y=targets[mask]
    spoken=y>=5
    if not len(y) or not spoken.any():raise ValueError('Complete teacher requires valid spoken targets')
    return dict(hidden_maximum_absolute_error=float((hidden.float()-stream_hidden.float()).abs().max()),
        logits_maximum_absolute_error=float((logits.float()-stream_logits.float()).abs().max()),
        valid_logits_maximum_absolute_error=float((z-s).abs().max()),
        valid_logits_mean_absolute_error=float((z-s).abs().mean()),
        valid_targets=len(y),spoken_targets=int(spoken.sum()),
        valid_argmax_disagreements=int((z.argmax(-1)!=s.argmax(-1)).sum()),
        spoken_argmax_disagreements=int((z[spoken].argmax(-1)!=s[spoken].argmax(-1)).sum()),
        batch_target_ce=float(F.cross_entropy(z,y)),stream_target_ce=float(F.cross_entropy(s,y)),
        batch_spoken_target_ce=float(F.cross_entropy(z[spoken],y[spoken])),
        stream_spoken_target_ce=float(F.cross_entropy(s[spoken],y[spoken])),
        batch_to_stream_kl=float(F.kl_div(s.log_softmax(-1),z.softmax(-1),reduction='batchmean')))


def run(lm,old,new,args):
    import torch
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_execution import snapshot
    validate(args)
    if len(old)!=25 or len(new)!=24:raise ValueError('Every approved old25 and new24 teacher is required')
    baseline=snapshot(lm)
    if 'torch.bfloat16' not in baseline['parameter_dtypes']:raise ValueError('Actual BF16 model required')
    before=tensor_hashes(lm.named_parameters());grad_flags={n:p.requires_grad for n,p in lm.named_parameters()}
    lm.requires_grad_(False).eval();records=[];manifests=[]
    try:
        with torch.no_grad():
            for folder,batches in [(args.actual_native_teacher_preflight,old),(args.native_v2_preflight,new)]:
                path=folder/'native-teacher-data.json';manifest=json.loads(path.read_text());manifests.append(dict(path=str(path),sha256=sha256(path)))
                if len(manifest['cases'])!=len(batches):raise ValueError('Incomplete approved manifest')
                for inputs,mask,targets,teacher in batches:
                    binding=next(c for c in manifest['cases'] if c['teacher']['id']==teacher['id'])
                    capture=Path(binding['capture']);report=capture/'teacher-engine-input-audit.json'
                    if sha256(report)!=binding['capture_report_sha256']:raise ValueError('Approved call trace changed')
                    trace=json.loads(report.read_text());calls=trace['calls'];call_partition(calls,inputs.shape[-1])
                    h,z=lm.forward_codes(inputs);hs=[];zs=[]
                    with lm.streaming(1):
                        for call in calls:
                            start,n=call['start'],call['frames'];sh,sz=lm.forward_codes(inputs[:,:,start:start+n])
                            hs.append(sh.detach().clone());zs.append(sz.detach().clone())
                    sh=torch.cat(hs,dim=1);sz=torch.cat(zs,dim=2)
                    row=dict(teacher=teacher,call_trace_sha256=sha256(report),native_call_count=len(calls),
                        input_frames=inputs.shape[-1],**compare(h,z,sh,sz,mask,targets))
                    records.append(row);print(json.dumps(row),flush=True)
                    del h,z,sh,sz,hs,zs
    finally:
        for n,p in lm.named_parameters():p.requires_grad_(grad_flags[n])
    after=snapshot(lm)
    if before!=tensor_hashes(lm.named_parameters()) or baseline!=after:raise RuntimeError('Diagnostic changed weights or execution policy')
    if len(records)!=49 or len({r['teacher']['id'] for r in records})!=49:raise RuntimeError('Incomplete49 diagnostic')
    result=dict(schema='dl-native-bf16-stream-batch49-diagnostic-v1',diagnostic_only=True,
        all49_complete=True,optimizer_updates=0,all_parameters_unchanged=True,execution_policy_unchanged=True,
        native520_checkpoint_sha256=sha256(args.warm_start_proof/'adapters.safetensors'),
        execution=after,approved_manifests=manifests,cases=records,
        original_reference_threshold=.001,original_reference_approvals_unchanged=True,
        original_all49_fp32_fp64_approval_is_not_a_bf16_equivalence_claim=True,
        training_approved=False,training_extension_or_adoption_approved=False,
        causal_and_depth_reference_gates_unchanged=True,
        scope='Actual BF16/FP32-adapter MATH eager temporal forward; native recorded call partition versus batched exact same delayed integer inputs. No LMGen sampling/graphs, no acoustic-depth parity or behavioral qualification claim.')
    (args.out/'bf16-stream49-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)
