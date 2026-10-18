"""New role-reversal holdouts, declared before the fixed1599-update model.

The TRAIN collector never imports this file. No fixture expectation enters the
model prompt; the actual latest request selects the other real task identity.
"""
import argparse
import copy
import json
from pathlib import Path
from data.synthetic.filter import norm_words
from training.control.state_curriculum import new_scenes,state_world
from evals.idempotent_reasoning_worker import run_idempotent_reasoning_worker,require_idempotent_delivery


def cases():
    rows=[]
    kinds=('resume-active','resume-finished','resume-cancelled','resume-duplicates','cancel-new-answer','resume-keep-work')
    for base in new_scenes():
        if base['split']!='heldout' or base['kind'] not in kinds:continue
        kind=base['kind'];scene=copy.deepcopy(base)
        if kind=='cancel-new-answer':
            user=f'Cancel delivery of the {base["other"]} answer, keeping the {base["target"]} unchanged. Replace your directions by giving only the answer to twelve minus five.'
        elif kind=='resume-keep-work':
            user=f'Carry on with the same {base["item"]} explanation. Keep every calculation result delivery active; cancel nothing.'
        else:
            user=f'Continue the paused {base["item"]} explanation without restarting it. Cancel only delivery of the {base["other"]} result; leave the {base["target"]} alone.'
        scene.update(id='roleflip-'+base['id'],user=user)
        rows.append(scene)
    assert len(rows)==24
    return rows


def evaluate(executor,folder):
    from evals.application_capabilities import ApplicationCapabilityExecutor
    folder.mkdir();rows=[];profile=ApplicationCapabilityExecutor(executor)
    for scene in cases():
        context,world,revision,pending,options=state_world(scene)
        before=copy.deepcopy(world.tasks);target,other,duplicate=scene['task_ids'];kind=scene['kind']
        expected=[] if kind=='resume-keep-work' else [other]
        row=dict(id=scene['id'],scene=scene,context=context,actual_world_before=before,passed=False)
        try:
            result=run_idempotent_reasoning_worker(context,world,'latest',executor.model_id,revision,'state-aware',pending,profile,**options)
            changed=sorted(k for k,v in world.tasks.items() if v['cancelled'] and not before[k]['cancelled'])
            checks=dict(exact_requested_cancellation=changed==expected,
                all_other_tasks_preserved=all(world.tasks[k]==v for k,v in before.items() if k not in expected and k!='latest'))
            if kind=='cancel-new-answer':
                checks.update(route=result['route']=='cancel',new_answer_only=norm_words(result['content'])==norm_words('seven'),old_answer_retired=result['consume_pending_speech_ids']==['retained-directions'])
            else:
                checks.update(route=result['route']=='resume',same_generator=result.get('resume_target')==options['resumable_speech'],no_new_words=result['content']=='',speech_preserved=result['consume_pending_speech_ids']==[])
            if result.get('resume_after_idempotent_cancellation'):require_idempotent_delivery(result['receipts'],world.tasks)
            row.update(result=result,checks=checks,passed=all(checks.values()))
        except Exception as error:row.update(error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
        row['actual_world_after']=world.tasks;rows.append(row);(folder/(scene['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    report=dict(cases=24,passed=sum(r['passed'] for r in rows),results=rows,do_not_train=True,actual_external_requests=0,overall_qualified=False)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n');return {k:v for k,v in report.items() if k!='results'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','kernel-assets','candidate','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    from training.control.head_adapter import parameter_hashes
    from evals.context_candidate import attach
    from evals.moshi_model_download import sha256
    x=Qwen35ResponseExecutor(a.assets,kernel_assets=a.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    warm_long_prefill(x);source=dict(x.model.named_parameters());before=parameter_hashes(source)
    baseline=evaluate(x,a.out/'baseline')
    candidate=attach(x,a.candidate,a.assets,1599)
    adapted=evaluate(x,a.out/'candidate')
    if before!=parameter_hashes(source):raise ValueError('Role holdout changed pretrained source parameters')
    report=dict(schema='dl-state-role-holdout-v1',baseline=baseline,candidate=adapted,candidate_proof=candidate,
        source_parameters_unchanged=True,optimizer_steps=0,actual_external_requests=0,
        code_sha256=sha256(Path(__file__)),do_not_train=True,production_adopted=False,overall_qualified=False)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
