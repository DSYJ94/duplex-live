"""Summarize recorded training objectives without loading a model or changing gates.

The 154 retained terms repeat every update. Native49/control40 teacher partitions
rotate, so their combined step totals must not be presented as a convergence curve.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from evals.context_native_complete_batch import validate_groups


def summarize(rows):
    if not rows or len(rows)>8 or [r['step']for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('Contiguous recorded updates from one bounded8 experiment required')
    result=[];membership=None
    for row in rows:
        groups=row['complete_balanced_microbatches'];validate_groups(groups)
        current={name:[(x['id'],x.get('kind'),x['weight'])for x in values]for name,values in groups.items()}
        if membership is None:membership=current
        if current!=membership:raise ValueError('Fixed154 membership or weights changed across updates')
        if any(not math.isfinite(x['loss'])for values in groups.values()for x in values):
            raise ValueError('Nonfinite recorded objective')
        weighted={name:math.fsum(x['weight']*x['loss']for x in values)for name,values in groups.items()}
        if not math.isclose(weighted['actual'],row['future_text_ce'],rel_tol=1e-6,abs_tol=1e-7):
            raise ValueError('Legacy logged actual loss differs from recorded full28 objective')
        native=row['native_stream_teacher_microbatch'];control=row['control40_microbatch']
        if len(native)not in(6,7)or len(control)!=5 or len({x['family']for x in control})!=5:
            raise ValueError('Complete selected native/control partition required')
        native_loss=math.fsum(x['weight']*x['balanced_spoken_clock_ce']for x in native)
        text=math.fsum(x['weight']*x['text_ce']for x in control)
        acoustic=math.fsum(x['weight']*x['acoustic_ce']for x in control)
        fixed=math.fsum(weighted.values());total=fixed+native_loss+text+acoustic
        norm=row['gradient_norm_before_clip']
        if not all(math.isfinite(x)for x in (fixed,native_loss,text,acoustic,total,norm))or norm<=0:
            raise ValueError('Finite recorded losses and positive gradient norm required')
        result.append(dict(step=row['step'],losses_measured_before_this_update=True,
                           fixed154_weighted_groups=weighted,fixed154_weighted_total=fixed,
                           selected_native49_weighted_loss=native_loss,
                           selected_control40_weighted_text_loss=text,
                           selected_control40_weighted_acoustic_loss=acoustic,
                           selected_total_weighted_objective=total,
                           selected_native_teacher_ids=[x['id']for x in native],
                           selected_control_teacher_ids=[x['id']for x in control],
                           gradient_norm_before_clip=norm))
    first=result[0]['fixed154_weighted_total'];last=result[-1]['fixed154_weighted_total']
    return dict(schema='dl-native-control40-recorded-loss-summary-v1',recorded_updates=len(result),
                complete8_recorded=len(result)==8,steps=result,
                fixed154_loss_delta=None if len(result)<2 else last-first,
                fixed154_relative_reduction=None if len(result)<2 or first==0 else (first-last)/first,
                fixed154_membership_exact_across_recorded_updates=True,
                comparable_across_steps='Only the fixed154 objective repeats the same terms and weights every update.',
                rotating_partition_totals_are_not_a_convergence_curve=True,
                convergence_established=False,independent_restore_established=False,
                behavioral_improvement_established=False,training_extension_or_adoption_approved=False,
                interpretation='Training losses describe fit on TRAIN teachers; validation, native restore and behavioral tests remain separate requirements.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raw=a.log.read_bytes();rows=[]
    for line in raw.decode().splitlines():
        try:value=json.loads(line)
        except ValueError:continue
        if isinstance(value,dict)and 'complete_balanced_microbatches'in value:rows.append(value)
    report=summarize(rows)
    report.update(source_log=str(a.log.resolve()),source_log_sha256=hashlib.sha256(raw).hexdigest(),
                  analysis_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  model_or_GPU_loaded=False)
    with a.out.open('x')as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k!='steps'}))


if __name__=='__main__':main()
