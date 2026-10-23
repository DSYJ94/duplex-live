"""Diagnostic geometry from every recorded control40 case; no new model work.

Use ||text+increment||, ||text|| and ||increment|| from the actual independent
gradient report. The increment is the recorded FP32 accumulation difference,
including rounding, and its text-head delta is exactly zero. This is teacher-local
geometry, not retained154-vs-control interference, an Adam update or qualification.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from portable.context.native_control_copyback import verify


def cosine_from_norms(text,total,increment):
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in (text,total,increment)) or text<=0 or total<0 or increment<=0:
        raise ValueError('Finite nonzero component norms required')
    scale=max(text,total,increment);a=text/scale;b=total/scale;d=increment/scale
    if b>a+d+1e-12 or b<abs(a-d)-1e-12:raise ValueError('Norms violate the vector triangle inequality')
    denominator=2*a*d
    if denominator==0:raise ValueError('Insufficient floating-point resolution for component angle')
    cosine=math.fsum((b*b,-a*a,-d*d))/denominator
    if not -1.-1e-9<=cosine<=1.+1e-9:raise ValueError('Reported norms cannot resolve a valid gradient angle')
    return cosine


def analyze(report,*,allow_partial_diagnostic=False):
    cases=report['cases'];n=len(cases)
    if (report['schema']!='dl-native-control40-gradient-readiness-v1' or not 1<=n<=40
            or len({r['id'] for r in cases})!=n or report['optimizer_updates']!=0
            or report['checkpoint_written'] is not False or report['trainable_parameters']!=1894400):
        raise ValueError('Exact recorded zero-update gradient source required')
    complete=report['complete40'] is True and report['passed'] is True and n==40
    if not complete and not allow_partial_diagnostic:
        raise ValueError('Partial evidence requires explicit diagnostic-only mode; never approval')
    rows=[]
    for r in cases:
        text=r['text_gradient'];both=r['combined_gradient']
        if (r['passed'] is not True or r['exact_native_streamed_text_outputs'] is not True
                or r['exact_native_acoustic_outputs'] is not True
                or both['acoustic_text_head_gradient_delta_exactly_zero'] is not True
                or any(x['all26_gradients_present_and_finite'] is not True or x['frozen_parameter_gradients']!=0
                       for x in (text,both))):raise ValueError('All recorded cases must preserve exact output and gradient ownership')
        a=text['combined_gradient_norm'];b=both['combined_gradient_norm'];d=both['acoustic_temporal_gradient_delta_norm']
        cosine=cosine_from_norms(a,b,d)
        rows.append(dict(id=r['id'],family=r['family'],text_gradient_norm=a,
            accumulated_gradient_norm=b,acoustic_increment_norm=d,acoustic_increment_to_text_norm_ratio=d/a,
            text_vs_acoustic_increment_cosine=cosine,opposing_local_dot_product=cosine<0,
            increment_includes_actual_FP32_accumulation_rounding=True))
    return dict(schema='dl-native-control40-teacher-local-gradient-geometry-v1',at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=report['source_commit'],source_gradient_gate_complete=complete,
        case_count=n,expected_case_count=40,all_recorded_cases_included=True,
        family_counts=dict(Counter(r['family'] for r in rows)),partial_diagnostic_only=not complete,
        median_acoustic_increment_to_text_norm_ratio=statistics.median(r['acoustic_increment_to_text_norm_ratio'] for r in rows),
        minimum_cosine=min(r['text_vs_acoustic_increment_cosine'] for r in rows),
        median_cosine=statistics.median(r['text_vs_acoustic_increment_cosine'] for r in rows),
        maximum_cosine=max(r['text_vs_acoustic_increment_cosine'] for r in rows),
        opposing_local_case_count=sum(r['opposing_local_dot_product'] for r in rows),cases=rows,
        diagnostic_only=True,training_approved=False,behavior_qualified=False,
        original154_or_old49_gradient_conflict_measured=False,
        cross_example_accumulation_or_Adam_preconditioning_measured=False,
        cause_of_previous7590052_failures_established=False,model_modified=False,new_optimizer_updates=0,
        new_GPU_or_API_calls=0,limitation='Local text-vs-acoustic-increment geometry on current control40 teachers at native520. Not an adoption gate, full retained-objective comparison, previous-pilot ablation or behavioral evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--inventory-sha256',required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-partial-diagnostic',action='store_true')
    a=p.parse_args();verify(a.snapshot,a.inventory_sha256)
    path=a.snapshot/'native-control40-precise-pilot-v2/gradient/gradient.json'
    r=analyze(json.loads(path.read_text()),allow_partial_diagnostic=a.allow_partial_diagnostic)
    r.update(verified_snapshot=str(a.snapshot),verified_inventory_sha256=a.inventory_sha256,
             gradient_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({k:r[k] for k in ('case_count','partial_diagnostic_only','opposing_local_case_count','median_cosine','median_acoustic_increment_to_text_norm_ratio')}))
