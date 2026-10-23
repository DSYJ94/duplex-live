"""Plan all40 portable fresh cases only after the complete development review.

Uses actual checkpoint directories and an explicit serial single-GPU resource
cohort. This is a preflight, not an inference collector or qualification stamp.
No Slurm identities, scoring answers, TRAIN recorder or optimizer are supplied.
"""
import argparse
import json
from pathlib import Path

from evals.moshi_model_download import sha256
from evals.native_fresh_session import PLAN,PLAN_SHA256,PROFILE,ROOT


def cases(proof,baseline):
    proof=Path(proof).resolve();baseline=Path(baseline).resolve()
    if proof==baseline or baseline.name!='context-training-proof-7510331':
        raise ValueError('Distinct actual candidate and exact native520 comparator required')
    plan=json.loads(PLAN.read_text())
    if (sha256(PLAN)!=PLAN_SHA256 or len(plan['cases'])!=20
            or plan['do_not_train'] is not True or plan['training_approved'] is not False):
        raise ValueError('Complete unchanged sealed20 evaluation plan required')
    rows=[]
    for index in range(40):
        case=plan['cases'][index%20]
        rows.append(dict(index=index,question_index=index%20,scenario=case['id'],
            role='baseline520' if index<20 else 'candidate-control40',
            proof=str(baseline if index<20 else proof),
            kind={'ordinary':'simple','quiet':'simple','pause':'simple'}.get(case['kind'],case['kind']),
            backend=case['backend'],minimum_lookup_delay_s=case.get('minimum_actual_lookup_delay_s',0.),
            native_asr_placement='shared-gpu-evaluation',assigned_cuda_devices=['0'],required_gpus=1))
    if len({r['scenario'] for r in rows[:20]})!=20 or any(
            rows[i]['scenario']!=rows[i+20]['scenario'] or rows[i]['backend']!=rows[i+20]['backend']
            or rows[i]['minimum_lookup_delay_s']!=rows[i+20]['minimum_lookup_delay_s'] for i in range(20)):
        raise ValueError('Every sealed case must appear exactly once for each model')
    return rows


def preflight(proof,review,data,quality,out):
    from evals.native_fresh_review import validate
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.native_fresh_source_reapproval import approved_inputs,PROFILE as approved_profile
    proof=Path(proof).resolve();review=Path(review).resolve()
    if ROOT.resolve() not in review.parents:
        raise ValueError('Behavior review must be committed inside this immutable source')
    # Review/independent restore must be checked BEFORE fresh waveform/quality reads.
    validated=validate(review,proof,proof)
    if (validated['schema']!='dl-native-control40-development-review-v1'
            or validated['native_asr_placement']!='shared-gpu-evaluation'):
        raise ValueError('Explicit complete portable same-resource native review required')
    identity,parent=verify_checkpoint(proof)
    baseline=Path(parent['warm_start']['proof'])
    base_identity,base=verify_checkpoint(baseline)
    if (parent['warm_start']!=base_identity or approved_profile!=PROFILE
            or parent['voice_sha256']!=base['voice_sha256'] or parent['data_sha256']!=base['data_sha256']
            or identity['native_protocol']!=base_identity['native_protocol']):
        raise ValueError('Paired original source data, voice, native protocol and parent must remain identical')
    rows=cases(proof,baseline)
    manifest=approved_inputs(Path(data),Path(quality))
    if (len(manifest['rows'])!=20 or {r['scenario_id'] for r in manifest['rows']}
            !={r['scenario'] for r in rows}):
        raise ValueError('Complete independently reapproved20 source inputs required')
    result=dict(schema='dl-native-control40-fresh-paired40-preflight-v1',complete40=True,
        evaluation_only=True,training_approved=False,checkpoint_adopted=False,
        candidate_proof=str(proof),baseline_proof=str(baseline),
        candidate_identity=identity,baseline_identity=base_identity,
        behavior_review_sha256=sha256(review),source_quality_sha256=sha256(Path(quality)),
        plan_sha256=PLAN_SHA256,source_profile=PROFILE,cases=rows,
        batches=[[i] for i in range(40)],gpu_mode='serial-one-gpu',maximum_gpus=1,
        scheduler_array=False,no_passing_subset=True,
        baseline_and_candidate_same_questions_voice_seed_sampler=True,
        scoring_references_never_passed_to_model=True,
        resource_timing_comparable_to_two_gpu=False,
        actual_inference_executed=False,behavior_or_adoption_approved=False)
    with Path(out).open('x') as handle:
        json.dump(result,handle,indent=2);handle.write('\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('proof','review','data','quality','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    preflight(args.proof,args.review,args.data,args.quality,args.out)


if __name__=='__main__':main()
