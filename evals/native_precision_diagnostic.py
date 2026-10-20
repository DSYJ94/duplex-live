"""Read-only complete2x2 checkpoint/compute-precision comparison, original ordinary4."""
from pathlib import Path
from collections import Counter

ORDINARY=tuple('recovery-train-%03d'%i for i in range(12,16))
CANDIDATE='7585424'


def case(index,no_tf32=False):
    if type(index) is not int or not 0<=index<16:raise ValueError('Complete sixteen-case precision comparison required')
    if type(no_tf32) is not bool:raise ValueError('TF32 diagnostic requires an exact boolean')
    arm=index%4;child=bool(arm%2)
    row=dict(index=index,scenario=ORDINARY[index//4],role='complete-batch8' if child else 'native520',
        checkpoint_job=CANDIDATE if child else '7510331',precision='fp32-lm' if arm>=2 else 'original-bf16',
        text_temperature=.7,acoustic_temperature=.8,seed=42424242,optimizer_steps=0,diagnostic_only=True,
        training_approved=False,native_quality_or_extension_approved=False,diagnostic_outputs_must_not_be_trained_on=True)
    if no_tf32:row.update(lm_matmul_tf32=False,tf32_scope='LM forward_codes and forward_depformer only; restore global policy before Mimi')
    return row


def policy(args):
    row=case(args.native_precision_diagnostic_case,getattr(args,'native_precision_no_tf32',False))
    if (not args.native_precision_diagnostic or args.scenario!=row['scenario'] or args.expected_kind!='simple'
            or args.backend_mode!='observe-only' or not args.train_history or not args.adapter_proof
            or Path(args.adapter_proof).name!='context-training-proof-'+row['checkpoint_job']
            or args.adapter_disabled or args.adapter_zero or args.native_text_temperature!=.7
            or args.voice!='accepted-synthetic' or args.warm_audio or args.ret_logit_bias or args.pending_status
            or args.persona_profile!='recorded-records' or args.lookup_delivery_delay_s
            or any(getattr(args,n,False) for n in ('native_sampling_diagnostic','native_seen_retention_diagnostic','native_persona_diagnostic','native_ancestor_diagnostic'))
            or any(getattr(args,n,None) is not None for n in ('native_persona_diagnostic_case','native_ancestor_diagnostic_case','layout_audit'))
            or getattr(args,'math_attention',False) or getattr(args,'backend_reference_profile','answer-only')!='answer-only'
            or getattr(args,'event_log_mode','immediate')!='immediate'):
        raise ValueError('Only the declared checkpoint and compute precision may vary in this locked16 diagnostic')
    return dict(row,mode='native-checkpoint-precision16-no-tf32-diagnostic-v1' if getattr(args,'native_precision_no_tf32',False) else 'native-checkpoint-precision16-diagnostic-v1',forced_reference_words=False,
        source_words_audio_persona_voice_and_checkpoint_bytes_unchanged=True,
        precision_scope='LM tensor dtype comparison; TF32 disabled inside native LM methods only, original global policy restored before codec; no optimizer' if getattr(args,'native_precision_no_tf32',False) else 'FP32 LM tensors versus unchanged original BF16, original TF32 flags retained so Mimi codec arithmetic stays unchanged; no optimizer')


def scoped_lm_no_tf32(lm):
    """Choose LM matmul kernels before native graphs, leaving Mimi's policy intact."""
    import torch
    from functools import wraps
    names=('forward_codes','forward_depformer')
    if not all(callable(getattr(lm,n,None)) for n in names):raise ValueError('Verified native LM methods required')
    evidence=dict(scope=list(names),wrapper_calls={},first_call_execution={},
        call_count_scope='Python eager/warmup/graph capture calls, not CUDA graph replay frame counts',
        restored_before_return=True,mimi_global_policy_unchanged=True)
    for name in names:
        original=getattr(lm,name)
        def bind(original,name):
            @wraps(original)
            def call(*args,**kwargs):
                previous=torch.backends.cuda.matmul.allow_tf32
                previous_precision=torch.get_float32_matmul_precision()
                torch.backends.cuda.matmul.allow_tf32=False
                try:
                    if torch.backends.cuda.matmul.allow_tf32:raise RuntimeError('LM TF32 disable failed')
                    evidence['wrapper_calls'][name]=evidence['wrapper_calls'].get(name,0)+1
                    if name not in evidence['first_call_execution']:
                        evidence['first_call_execution'][name]=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,float32_precision=torch.get_float32_matmul_precision())
                    return original(*args,**kwargs)
                finally:
                    torch.set_float32_matmul_precision(previous_precision)
                    if torch.backends.cuda.matmul.allow_tf32!=previous:raise RuntimeError('LM TF32 wrapper did not restore original global precision')
            return call
        setattr(lm,name,bind(original,name))
    return evidence


def convert(lm,precision):
    import torch
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_execution import snapshot
    if precision not in ('original-bf16','fp32-lm'):raise ValueError('Unknown locked precision')
    if any(p.requires_grad for p in lm.parameters()):raise ValueError('Precision comparison must remain frozen')
    original=snapshot(lm);all_tensors=list(lm.named_parameters())+[('buffer:'+n,t) for n,t in lm.named_buffers()]
    dtypes={n:t.dtype for n,t in all_tensors};devices={n:t.device for n,t in all_tensors}
    if not any(t.dtype==torch.bfloat16 for _,t in all_tensors):raise ValueError('Original native BF16 model required')
    if precision=='fp32-lm':
        hashes=tensor_hashes(all_tensors)
        lm._apply(lambda t:t.float() if t.dtype in (torch.bfloat16,torch.float16) else t)
        after=list(lm.named_parameters())+[('buffer:'+n,t) for n,t in lm.named_buffers()]
        if set(dtypes)!={n for n,_ in after} or any(t.device!=devices[n] for n,t in after):raise ValueError('Precision conversion changed structure/device')
        if tensor_hashes((n,t.to(dtype=dtypes[n])) for n,t in after)!=hashes:raise ValueError('Precision conversion changed checkpoint numeric values')
        if any(t.dtype!=(torch.float32 if dtypes[n] in (torch.bfloat16,torch.float16) else dtypes[n]) for n,t in after):raise ValueError('Unexpected dtype conversion')
    if any(p.requires_grad for p in lm.parameters()):raise ValueError('Conversion unfroze native weights')
    current=snapshot(lm)
    return dict(precision=precision,applied_before_native_graphs=True,conversion_applied=precision=='fp32-lm',all_weights_frozen=True,
        numeric_values_roundtrip_exact=True,checkpoint_bytes_modified=False,optimizer_steps=0,
        original_execution=original,actual_execution=current,mimi_codec_unchanged=True,tf32_flags_left_unchanged=True,
        fp32_roundtrip_checks_performed=precision=='fp32-lm')


def transform(declared):
    def apply(lm):
        result=convert(lm,declared['precision'])
        if declared.get('lm_matmul_tf32') is False:
            result['scoped_lm_no_tf32']=scoped_lm_no_tf32(lm)
            result.update(tf32_flags_left_unchanged=False,global_tf32_policy_restored_between_lm_calls=True)
        return result
    apply.native_precision_diagnostic=True
    return apply


def verify_engine(engine,declared,adapter):
    import torch
    from evals.context_execution import snapshot
    actual=snapshot(engine.lm)
    if declared['precision']=='fp32-lm':
        if set(actual['parameter_dtypes'])!={'torch.float32'}:raise RuntimeError('Native engine changed declared FP32 LM tensor policy')
    elif actual['parameter_dtypes']!=adapter['runtime_precision_diagnostic']['original_execution']['parameter_dtypes']:raise RuntimeError('Original BF16 control changed')
    adapter['runtime_precision_diagnostic']['execution_after_engine_construction']=actual
