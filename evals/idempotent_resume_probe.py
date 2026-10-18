"""Frozen actual-tool comparison; no training, speech, or external API calls."""
import argparse
import copy
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def blocked(question):
    raise RuntimeError('No external request allowed in this fixture trial')


def check(result, world, targets, active, pending, resume):
    return dict(route=result['route']=='resume', no_new_words=result['content']=='',
        same_generator=result.get('resume_target')==resume, speech_preserved=result['consume_pending_speech_ids']==[],
        exact_cancellation=sorted(k for k,v in world.tasks.items() if v['cancelled'])==sorted(active),
        no_claim_of_remote_abort=all(not r.get('remote_execution_aborted', False) for r in result['receipts']))


def focused(executor, worker, runs, folder, original):
    from evals.duplex_reasoning_worker import ReasoningWorld
    from evals.instruction_contract_probe import resume_fixture
    from evals.duplex_actual_controller import capture
    from evals.failed_controller_capture import capture_failure
    folder.mkdir(); cases=[]
    failed=runs/'duplex-live-transport-7525920/live-session-0/turn-3-failure.json'
    f=json.loads(failed.read_text()); meta=f['task']; task=meta['task_id']
    cases.append(dict(id='actual-closed',world=copy.deepcopy(f['actual_world_tasks']),task=task,
        context=copy.deepcopy(f['actual_world_tasks'][task]['context']),pending=meta['pending_speech'],resume=meta['resume_target'],
        expected_active=[],targets=['turn-1'],source=capture_failure(failed)[1]))
    payload,source=capture(runs/'duplex-live-transport-7524610','turn-3'); state=payload['local_worker_state']
    context=json.loads(payload['input']); tasks=copy.deepcopy(state['available']['tasks'])
    tasks['probe-current']=dict(revision=1,cancelled=False,closed=False,phase='pending',context=context)
    cases.append(dict(id='actual-active',world=tasks,task='probe-current',context=context,pending=state['pending_speech'],
        resume=state['resumable_speech'],expected_active=['turn-1'],targets=['turn-1'],source=source))
    for status in ('active','finished','unrelated-active'):
        for variant in range(3):
            scene,_=resume_fixture(status,variant,executor.model_id)
            cases.append(dict(id=f'{status}-{variant}',world=scene['actual_world_tasks'],task='latest',context=scene['context'],
                pending=scene['pending_speech'],resume=scene['resume_target'],expected_active=['blue-count'] if status=='active' else [],targets=['blue-count']))
    rows=[]
    for scene in cases:
        world=ReasoningWorld({},reasoner=blocked);world.tasks=copy.deepcopy(scene['world'])
        row=dict(id=scene['id'],scene=scene,passed=False)
        try:
            result=worker(copy.deepcopy(scene['context']),world,scene['task'],executor.model_id,
                world.tasks[scene['task']]['revision'],'state-aware',copy.deepcopy(scene['pending']),executor,
                resumable_speech=scene['resume'],bound_resume_target=True,allow_cancel_resume=True)
            checks=check(result,world,scene['targets'],scene['expected_active'],scene['pending'],scene['resume'])
            row.update(result=result,checks=checks,passed=all(checks.values()))
        except Exception as error:
            row.update(error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
        row['actual_world_after']=world.tasks
        if original and scene.get('source'):
            evidence=(row.get('result',{}).get('events') or [{}])[0].get('local_inference') or (row.get('evidence') or {}).get('local_generation',{})
            row['historical_input_length_exact']=evidence.get('input_tokens')==scene['source']['original_input_tokens']
            row['historical_raw_output_exact']=evidence.get('raw_text')==scene['source']['original_raw_text']
            if not row['historical_input_length_exact'] or not row['historical_raw_output_exact']:
                raise ValueError('Original actual input/output reconstruction changed')
        rows.append(row);(folder/(scene['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    (folder/'report.json').write_text(json.dumps(dict(cases=len(rows),passed=sum(r['passed'] for r in rows),results=rows),indent=2)+'\n')
    return rows


def paired(executor,worker,folder):
    from training.control.paired_curriculum import specifications,payload,assess
    folder.mkdir();rows=[]
    for scene in [s for s in specifications() if s['split']=='heldout']:
        row=dict(id=scene['id'],scene=scene,passed=False)
        try:
            request,world,revision,options=payload(scene,executor.model_id);world.reasoner=blocked
            if scene['kind']=='delegate':
                response=executor(request)
            else:
                result=worker(copy.deepcopy(scene['context']),world,'latest',executor.model_id,revision,'state-aware',
                    copy.deepcopy(scene['pending_speech']),executor,**options)
                event=result['events'][0]
                response=dict(id=event['response_id'],model=event['model'],status='completed',output=event['output'])
                row.update(result=result,actual_world_after=world.tasks)
                cancelled=sorted(k for k in scene['active_calculations'] if world.tasks[k]['cancelled'])
                if cancelled!=sorted(scene['expected']['cancelled']):
                    raise ValueError('Actual new-worker cancellations differ from declared targets')
            quality=assess(scene,response);row.update(response=response,quality=quality,passed=quality['passed'])
        except Exception as error:
            row.update(error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None))
        rows.append(row);(folder/(scene['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    (folder/'report.json').write_text(json.dumps(dict(cases=len(rows),passed=sum(r['passed'] for r in rows),results=rows),indent=2)+'\n')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('assets','kernel-assets','runs','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    from evals.duplex_reasoning_worker import run_reasoning_worker
    from evals.idempotent_reasoning_worker import run_idempotent_reasoning_worker
    from training.control.head_adapter import parameter_hashes
    executor=Qwen35ResponseExecutor(a.assets,kernel_assets=a.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    warm_long_prefill(executor);before=parameter_hashes(dict(executor.model.named_parameters()));reports={}
    for name,worker in [('original',run_reasoning_worker),('idempotent',run_idempotent_reasoning_worker)]:
        rows=focused(executor,worker,a.runs,a.out/(name+'-focused'),name=='original')
        controls=paired(executor,worker,a.out/(name+'-paired'))
        reports[name]=dict(focused_passed=sum(r['passed'] for r in rows),focused_cases=len(rows),paired_passed=sum(r['passed'] for r in controls),paired_cases=len(controls))
        print(json.dumps(dict(mode=name,summary=reports[name])),flush=True)
    if before!=parameter_hashes(dict(executor.model.named_parameters())):
        raise ValueError('Frozen model parameters changed')
    report=dict(schema='dl-idempotent-resume-probe-v1',summaries=reports,source_model=executor.proof,
        all_parameters_unchanged=True,optimizer_steps=0,actual_external_requests=0,outputs_repaired=False,
        user_keyword_routing=False,function_forcing=False,do_not_train=True,overall_qualified=False,
        production_media_receipt_guard_supported=False,production_adopted=False,owner_demo_changed=False,
        code_sha256=sha256(Path(__file__)),worker_sha256=sha256(Path('evals/idempotent_reasoning_worker.py')))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
