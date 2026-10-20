"""Complete40 paired fresh evaluation scheduling; never supplies model answers."""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256

PLAN=Path(__file__).with_name('fixtures')/'native-fulltask-qualification-v2-20261019.json'
ROOT=Path(__file__).resolve().parents[1]
PLAN_SHA256='6e44cac1d6bfa19c3453f813abaf4613651f8d051d2c2a0f3dd80c35c1121979'
PROFILE='owner-approved-turbo-spelling-v1'


def cases(candidate):
    if not isinstance(candidate,str) or not candidate.isdigit() or candidate=='7510331':
        raise ValueError('Distinct locked numeric candidate required')
    plan=json.loads(PLAN.read_text())
    if (sha256(PLAN)!=PLAN_SHA256 or len(plan['cases'])!=20
            or not plan['do_not_train'] or plan['training_approved']):
        raise ValueError('Complete sealed20 evaluation plan required')
    rows=[]
    for index in range(40):
        question=plan['cases'][index%20]
        rows.append(dict(index=index,question_index=index%20,scenario=question['id'],
            role='native520' if index<20 else 'candidate',
            checkpoint_job='7510331' if index<20 else candidate,
            kind={'ordinary':'simple','quiet':'simple','pause':'simple'}.get(question['kind'],question['kind']),
            backend=question['backend'],minimum_lookup_delay_s=question.get('minimum_actual_lookup_delay_s',0.),
            required_gpus=2 if question['backend']=='native-asr' else 1))
    return rows


def batches(rows):
    if [r['index'] for r in rows]!=list(range(40)):raise ValueError('Complete ordered40 required')
    result=[]
    for i in range(20):
        a,b=rows[i],rows[i+20]
        if a['scenario']!=b['scenario'] or a['required_gpus']!=b['required_gpus']:
            raise ValueError('Baseline/candidate routing must match')
        if a['required_gpus']==2:result.extend([[i],[i+20]])
        elif a['required_gpus']==1:result.append([i,i+20])
        else:raise ValueError('Only one-GPU observation or two-GPU native ASR supported')
    if sorted(i for batch in result for i in batch)!=list(range(40)):
        raise ValueError('A paired evaluation case was lost')
    return result


def preflight(candidate,review,data,quality,out):
    from evals.native_fresh_review import validate
    from evals.context_rollout_conditioning import verify_checkpoint
    from evals.native_fresh_source_reapproval import approved_inputs,PROFILE as approved_profile
    if ROOT.resolve() not in Path(review).resolve().parents:raise ValueError('Review must be deployed inside this immutable source')
    rows=cases(candidate);proof=Path(data).parent/('context-training-proof-'+candidate)
    baseline=proof.parent/'context-training-proof-7510331'
    # Full10 review precedes source waveform/quality reads and native outputs.
    validate(review,proof,baseline);validate(review,proof,proof)
    identity,parent=verify_checkpoint(proof);base_identity,base=verify_checkpoint(baseline)
    if (approved_profile!=PROFILE or parent['voice_sha256']!=base['voice_sha256']
            or parent['data_sha256']!=base['data_sha256']
            or identity['native_protocol']!=base_identity['native_protocol']):
        raise ValueError('Paired native model/voice/source protocol differs')
    manifest=approved_inputs(Path(data),Path(quality))
    if (len(manifest['rows'])!=20 or {r['scenario_id'] for r in manifest['rows']}
            !={r['scenario'] for r in rows}):
        raise ValueError('All20 independently reapproved source cases required')
    result=dict(schema='dl-native-fresh-paired40-session-preflight-v1',complete40=True,
        evaluation_only=True,training_approved=False,checkpoint_adopted=False,
        candidate=candidate,candidate_identity=identity,baseline_identity=base_identity,
        behavior_review_sha256=sha256(Path(review)),source_quality_sha256=sha256(Path(quality)),
        plan_sha256=PLAN_SHA256,source_profile=PROFILE,cases=rows,batches=batches(rows),
        scheduler_array=False,maximum_gpus=2,no_passing_subset=True,
        baseline_and_candidate_same_questions_voice_seed_sampler=True,
        scoring_references_never_passed_to_model=True)
    with Path(out).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result


def launch(candidate,index,allocation,step,devices,out):
    if type(index) is not int or not 0<=index<40:raise ValueError('Complete40 case index required')
    row=cases(candidate)[index];assigned=devices.split(',')
    if (not allocation.isdigit() or not step.isdigit() or len(assigned)!=row['required_gpus']
            or len(set(assigned))!=len(assigned) or any(not x for x in assigned)):
        raise ValueError('Actual allocation/step and exact distinct GPU assignment required')
    row.update(allocation_job=allocation,step=step,assigned_cuda_devices=assigned,
        scheduler_array=False,evaluation_only=True,training_approved=False,plan_sha256=PLAN_SHA256)
    with Path(out).open('x') as f:json.dump(row,f,indent=2);f.write('\n')
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command');sub.required=True
    p=sub.add_parser('preflight')
    for name in ('candidate','review','data','quality','out'):p.add_argument('--'+name,required=True)
    p=sub.add_parser('launch')
    for name in ('candidate','allocation','step','devices','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--index',type=int,required=True)
    a=ap.parse_args()
    if a.command=='preflight':preflight(a.candidate,a.review,a.data,a.quality,a.out)
    else:
        row=launch(a.candidate,a.index,a.allocation,a.step,a.devices,a.out)
        print('\n'.join(str(row[k]) for k in ('scenario','kind','backend','minimum_lookup_delay_s','checkpoint_job')))


if __name__=='__main__':main()
