"""Supplemental native-call teacher CE beside every original154 objective."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.context_native_bf16_stream_diagnostic import call_partition


def policy(data):
    rows=[dict(id=r['teacher']['id'],kind=r['teacher']['kind']) for r in data['cases']]
    if len(rows)!=49 or len({r['id'] for r in rows})!=49:raise ValueError('All49 distinct approved teachers required')
    counts=Counter(r['kind'] for r in rows)
    if set(counts)!={'private','simple','clarify','cancel'}:raise ValueError('All4 teacher kinds required')
    for r in rows:r['weight_per_epoch']=1/(4*counts[r['kind']]);r['weight_when_selected']=8*r['weight_per_epoch']
    return dict(mode='native-recorded-call-teacher-auxiliary-v1',teachers=rows,
        source_words_audio_receipts_folds_masks_unchanged=True,all_original154_terms_and_losses_unchanged=True,
        epoch_mean_weight=1.,schedule='Stride8; all49 exactly once across8 new updates; balanced4-kind epoch mean',
        truncation='Detach KV autograd history after each recorded call; forward cache values unchanged; not full temporal BPTT',
        implementation_sha256=sha256(Path(__file__)),
        gradient_implementation_sha256=sha256(Path(__file__).with_name('context_native_stream_gradient.py')),
        sampler_or_dtype_or_tf32_changed=False,excludes_prefix_span_user_audio_targets=True)


def bindings(args,old,new):
    batches=old+new
    if len(old)!=25 or len(new)!=24 or len({b[3]['id'] for b in batches})!=49:raise ValueError('Complete49 native input inventory required')
    result=[]
    for folder,group in [(args.actual_native_teacher_preflight,old),(args.native_v2_preflight,new)]:
        path=folder/'native-teacher-data.json';manifest=json.loads(path.read_text())
        if len(manifest['cases'])!=len(group):raise ValueError('Incomplete native input manifest')
        for batch in group:
            teacher=batch[3];binding=next(c for c in manifest['cases'] if c['teacher']['id']==teacher['id'])
            report=Path(binding['capture'])/'teacher-engine-input-audit.json'
            if sha256(report)!=binding['capture_report_sha256'] or binding['teacher']!=teacher:raise ValueError('Changed approved native call trace')
            trace=json.loads(report.read_text());calls=trace['calls'];call_partition(calls,batch[0].shape[-1])
            result.append((batch,calls))
    return result


def backward(lm,bound,declared,step):
    from evals.context_native_stream_gradient import capture
    if type(step) is not int or not 0<=step<8 or len(bound)!=49:raise ValueError('One bounded8 stage only')
    rows=[]
    for i in range(step,49,8):
        (x,m,y,teacher),calls=bound[i];p=declared['teachers'][i]
        if p['id']!=teacher['id'] or p['kind']!=teacher['kind']:raise ValueError('Changed native teacher order')
        r=capture(lm,x,m,y,calls,backward=True,weight=p['weight_when_selected'])
        rows.append(dict(id=teacher['id'],kind=teacher['kind'],weight=p['weight_when_selected'],**r))
    return rows


def exposure(steps,declared):
    if len(steps)!=8 or [s['step'] for s in steps]!=list(range(1,9)):raise ValueError('Exactly8 supplemental updates required')
    for i,step in enumerate(steps):
        actual=step['native_stream_teacher_microbatch'];expected=declared['teachers'][i::8]
        if len(actual)!=len(expected):raise ValueError('Incomplete native auxiliary partition')
        for row,p in zip(actual,expected):
            if row['id']!=p['id'] or row['kind']!=p['kind'] or row['weight']!=p['weight_when_selected'] or row['valid_targets']<1 or row['gradient_calls']<1 or row['prefix_span_user_audio_loss_targets']!=0:
                raise ValueError('Changed native auxiliary ownership, identity or weight')
    return True


def fingerprints(lm,bound):
    from evals.context_native_stream_gradient import capture
    result={}
    for (x,m,y,r),calls in bound:
        captured=capture(lm,x,m,y,calls)
        result[r['id']]=hashlib.sha256(json.dumps(captured['call_output_hashes'],separators=(',',':')).encode()).hexdigest()
        print(json.dumps(dict(native_stream_restore_fingerprint=r['id'],sha256=result[r['id']])),flush=True)
    if len(result)!=49:raise ValueError('Independent full49 stream restoration required')
    return result


def check_readiness(folder,declared,policy_value):
    import os
    p=folder/'native-stream49-gradient-readiness.json';r=json.loads(p.read_text())
    if (r['schema']!='dl-native-stream49-gradient-readiness-v1' or r['passed'] is not True
            or not r['all49_complete'] or not r['exact_grad_vs_no_grad_streamed_outputs']
            or not r['all_parameters_and_execution_unchanged'] or r['optimizer_updates']!=0 or r['checkpoint_written']
            or r['training_approved'] or r['training_extension_or_adoption_approved']
            or r['source_commit']!=os.environ['DL_TASK_COMMIT'] or r['policy']!=policy_value
            or r['native520_checkpoint_sha256']!=policy_value['parent']['checkpoint_sha256']
            or r['implementation_sha256']!=declared['gradient_implementation_sha256']
            or r['peak_allocated_bytes']>=80_000_000_000 or len(r['cases'])!=49
            or [x['teacher']['id'] for x in r['cases']]!=[x['id'] for x in declared['teachers']]
            or any(not x['exact_native_forward_hashes'] or not x['all_trainable_gradients_present_and_finite']
                or x['frozen_parameter_gradients']!=0 or not math.isfinite(x['combined_gradient_norm'])
                or x['combined_gradient_norm']<=0 for x in r['cases'])):
        raise ValueError('Independent complete49 exact native gradient readiness required')
    return sha256(p)
