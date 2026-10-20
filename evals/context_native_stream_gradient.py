"""Zero-update readiness for a supplemental native-call teacher objective.

This does not replace any legacy objective or authorize fitting. Gradients are
truncated at recorded call boundaries; cache VALUES and native calls stay exact.
No checkpointing/recomputation of the mutable streaming cache is permitted.
"""
import hashlib
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.context_native_bf16_stream_diagnostic import call_partition,validate as validate_base


def validate(args):
    if (not args.native_audio49_preflight_only or not args.native_audio49_audit
            or args.native_audio49_preflight or args.reload or args.resume_training
            or args.native_audio49_gradient_probe
            or getattr(args,'native_audio49_bf16_stream_diagnostic',False)
            or not args.native_audio49_parent_adam or not args.native_audio49_complete_batch
            or not getattr(args,'native_audio49_stream_auxiliary',False)
            or Path(args.warm_start_proof).name!='context-training-proof-7510331'):
        raise ValueError('Stream readiness requires the exact zero-update complete154 plus native49 recipe')


def coefficients(labels,mask):
    import torch
    if labels.dtype!=torch.int64 or mask.dtype!=torch.bool or labels.ndim!=1 or labels.shape!=mask.shape:
        raise ValueError('Exact one-dimensional approved labels and mask required')
    active=labels[mask]
    if not len(active) or (active<0).any():raise ValueError('Invalid approved targets')
    spoken=mask & (labels>=5);clock=mask & (labels<5)
    if not spoken.any() or not clock.any():raise ValueError('Both approved target groups required')
    result=torch.zeros_like(labels,dtype=torch.float32)
    result[spoken]=.5/spoken.sum()
    w=torch.where(labels[clock]==3,.5,1.)
    result[clock]=.5*w/w.sum()
    return result


def detach_caches(lm):
    """Detach only autograd history, after backward and before the next call."""
    import torch
    count=0
    for state in lm.get_streaming_state().values():
        cache=getattr(state,'kv_cache',None)
        if cache is None:continue
        if not isinstance(cache.cache,torch.Tensor) or cache.cache.ndim!=5:
            raise ValueError('Unknown native KV cache layout')
        # This keeps storage, values, dtype, device, offset, capacity and policy.
        cache.cache=cache.cache.detach();count+=1
    if not count:raise ValueError('No native temporal caches found')
    return count


def capture(lm,inputs,mask,targets,calls,backward=False,weight=1.):
    import torch
    from torch.nn import functional as F
    if type(weight) not in (int,float) or not math.isfinite(weight) or weight<=0:raise ValueError('Finite positive native auxiliary weight required')
    call_partition(calls,inputs.shape[-1]);coef=coefficients(targets,mask)
    total=0.;digests=[];gradient_calls=0;cache_count=None
    with lm.streaming(1):
        for call in calls:
            start,n=call['start'],call['frames'];m=mask[start:start+n]
            if call.get('phase') in ('prefix','span') and m.any():raise ValueError('Prefix or supplied span acquired loss targets')
            # Preserve the exact prefix, virtual span and real-frame partition.
            with torch.set_grad_enabled(backward):
                h,z=lm.forward_codes(inputs[:,:,start:start+n])
                if not torch.isfinite(h).all() or not torch.isfinite(z).all():raise ValueError('Nonfinite native outputs')
                hashed=[]
                for t in (h,z):
                    hashed.append(hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
                digests.append(hashed)
                if m.any():
                    y=targets[start:start+n][m];logits=z[0,0,m].float()
                    if (y>=logits.shape[-1]).any():raise ValueError('Target outside native vocabulary')
                    value=(F.cross_entropy(logits,y,reduction='none')*coef[start:start+n][m]).sum()
                    if not torch.isfinite(value):raise ValueError('Nonfinite streamed loss')
                    total+=float(value.detach())
                    if backward:(value*weight).backward();gradient_calls+=1
                    del value
            current=detach_caches(lm)
            if cache_count is not None and current!=cache_count:raise ValueError('Native cache membership changed')
            cache_count=current
            del h,z
    return dict(call_output_hashes=digests,balanced_spoken_clock_ce=total,
        gradient_calls=gradient_calls,native_call_count=len(calls),temporal_cache_count=cache_count,
        valid_targets=int(mask.sum()),spoken_targets=int((mask & (targets>=5)).sum()),
        prefix_span_user_audio_loss_targets=0,autograd_truncation='recorded native call boundaries only')


def run(lm,learned,old,new,args,policy_value):
    import torch
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_execution import snapshot
    validate(args)
    if len(old)!=25 or len(new)!=24 or len({b[3]['id'] for b in old+new})!=49:
        raise ValueError('All distinct old25 and new24 teachers required')
    before=tensor_hashes(lm.named_parameters());execution=snapshot(lm);rows=[];manifests=[]
    if 'torch.bfloat16' not in execution['parameter_dtypes']:raise ValueError('Actual BF16 model required')
    if {n for n,p in lm.named_parameters() if p.requires_grad}!=set(learned):raise ValueError('Changed adapter ownership')
    torch.cuda.reset_peak_memory_stats()
    for folder,batches in [(args.actual_native_teacher_preflight,old),(args.native_v2_preflight,new)]:
        path=folder/'native-teacher-data.json';manifest=json.loads(path.read_text());manifests.append(dict(path=str(path),sha256=sha256(path)))
        if len(manifest['cases'])!=len(batches):raise ValueError('Incomplete native manifest')
        for inputs,mask,targets,teacher in batches:
            binding=next(c for c in manifest['cases'] if c['teacher']['id']==teacher['id'])
            report=Path(binding['capture'])/'teacher-engine-input-audit.json'
            if sha256(report)!=binding['capture_report_sha256'] or binding['teacher']!=teacher:raise ValueError('Changed approved native trace')
            trace=json.loads(report.read_text());calls=trace['calls'];call_partition(calls,inputs.shape[-1])
            if int(mask.sum())!=teacher['text_targets'] or int((mask & (targets>=5)).sum())!=teacher['spoken_targets']:raise ValueError('Native teacher target coverage changed')
            for p in learned.values():p.grad=None
            reference=capture(lm,inputs,mask,targets,calls)
            current=capture(lm,inputs,mask,targets,calls,backward=True)
            # Exact hash parity against the same streamed computation, not a relaxed
            # tolerance against batch execution or a subset of passing teachers.
            if current['call_output_hashes']!=reference['call_output_hashes'] or current['balanced_spoken_clock_ce']!=reference['balanced_spoken_clock_ce']:
                raise ValueError('Gradient execution changed exact native streamed outputs')
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in learned.values()):raise ValueError('Missing/nonfinite adapter gradient')
            if any(p.grad is not None for n,p in lm.named_parameters() if n not in learned):raise ValueError('Frozen parameter gradient')
            norm=sum(float(p.grad.detach().double().square().sum()) for p in learned.values())**.5
            if not math.isfinite(norm) or not norm>0:raise ValueError('Dead native stream teacher gradient')
            row=dict(teacher=teacher,capture_report_sha256=sha256(report),capture_tensor_sha256=binding['capture_tensor_sha256'],
                exact_native_forward_hashes=True,combined_gradient_norm=norm,all_trainable_gradients_present_and_finite=True,
                frozen_parameter_gradients=0,**current)
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='call_output_hashes'}),flush=True)
            for p in learned.values():p.grad=None
    unchanged=before==tensor_hashes(lm.named_parameters()) and execution==snapshot(lm)
    peak=torch.cuda.max_memory_allocated()
    if len(rows)!=49 or not unchanged or peak>=80_000_000_000:raise ValueError('Incomplete49, changed model/policy, or exceeded80GB')
    result=dict(schema='dl-native-stream49-gradient-readiness-v1',passed=True,all49_complete=True,
        exact_grad_vs_no_grad_streamed_outputs=True,all_parameters_and_execution_unchanged=True,
        original_reference_threshold_unchanged=.001,batch_numerical_equivalence_claimed=False,
        original154_objectives_replaced=False,optimizer_updates=0,checkpoint_written=False,
        training_approved=False,training_extension_or_adoption_approved=False,
        truncation='Recorded native call boundaries; not full temporal BPTT and not a replacement for original objectives',
        source_commit=__import__('os').environ['DL_TASK_COMMIT'],execution=execution,policy=policy_value,
        native520_checkpoint_sha256=sha256(args.warm_start_proof/'adapters.safetensors'),
        implementation_sha256=sha256(Path(__file__)),peak_allocated_bytes=peak,
        approved_manifests=manifests,cases=rows)
    (args.out/'native-stream49-gradient-readiness.json').write_text(json.dumps(result,indent=2)+'\n')
