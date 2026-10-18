"""Two actual native proposals: original versus queued generated reply history.

The completed controller reply is real but has not started speaking. Proposals
are never dispatched. This development diagnostic uses no reference answers,
new supervision, model updates, API requests or physical playback claims.
"""
import argparse,copy,json
from pathlib import Path
from evals.moshi_model_download import sha256


def capture(root):
    from evals.duplex_reasoning_worker import ReasoningWorld,run_reasoning_worker
    suite=json.loads((root/'report.json').read_text());case=suite['results'][0];folder=root/case['scenario']
    first=json.loads((folder/'turn-1-backend.json').read_text());second=json.loads((folder/'turn-2-backend.json').read_text())
    original=second['result']['events'][0]['local_inference']
    assert first['result']['route']=='answer' and first['result']['receipts']==[]
    assert first['ready_s']<second['submitted_s']
    assert len(second['pending_speech'])==1 and second['pending_speech'][0]['id']=='turn-1'
    assert second['pending_speech'][0]['text']==first['result']['content']
    events=json.loads((folder/'events.json').read_text())
    assert not any(e['kind']=='speech_started' and e['task_id']=='turn-1' and e['at_s']<second['submitted_s'] for e in events)
    world=ReasoningWorld({});world.tasks=copy.deepcopy(case['world_tasks']);world.tasks['turn-2'].update(closed=False,phase='pending')
    context=copy.deepcopy(world.tasks['turn-2']['context'])
    class Capture:
        provider_name='fixture-only-payload-capture'
        def __call__(self,payload):
            self.payload=payload
            return dict(id='capture-only',status='completed',model='fixture-only',output=[dict(type='function_call',name='respond_to_user',arguments=json.dumps(dict(route='answer',content='Capture fixture only.',consume_pending_speech_ids=[])))])
    cap=Capture();run_reasoning_worker(context,world,'turn-2','Qwen/Qwen3.5-9B',world.tasks['turn-2']['revision'],'state-aware',second['pending_speech'],cap)
    assert cap.payload['local_worker_state']==original['compact_reply_action_protocol']
    assert json.loads(cap.payload['input'])==context and len(context)==2 and all(c['role']=='user' for c in context)
    actual=dict(role='assistant',text=first['result']['content'],control_source='actual-controller-generation',speech_task_id='turn-1',speech_delivery='generated-not-started',physical_hearing_confirmed=False)
    changed=copy.deepcopy(cap.payload);changed['input']=json.dumps([context[0],actual,context[1]])
    return cap.payload,changed,dict(source_report_sha256=sha256(root/'report.json'),first_backend_sha256=sha256(folder/'turn-1-backend.json'),second_backend_sha256=sha256(folder/'turn-2-backend.json'),original_input_tokens=original['input_tokens'],original_native_action=original['native_model_action'],actual_queued_reply=actual,actual_state_exact=True,changed_only_chronological_generated_reply=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','assets','kernel-assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    baseline,queued,source=capture(a.runs/'duplex-live-transport-7520301')
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    executor=Qwen35ResponseExecutor(a.assets,device='cuda:0',kernel_assets=a.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True)
    warm_long_prefill(executor);rows=[]
    for label,payload in [('baseline',baseline),('queued-generated-history',queued)]:
        result=executor(copy.deepcopy(payload));actual=result['local_inference']
        if label=='baseline' and (actual['input_tokens']!=source['original_input_tokens'] or actual['native_model_action']!=source['original_native_action']):raise RuntimeError('Original actual native proposal did not reproduce exactly')
        row=dict(variant=label,result=result);rows.append(row);(a.out/(label+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(dict(variant=label,native_action=actual['native_model_action'])),flush=True)
    report=dict(schema='dl-queued-generated-history-diagnostic-v1',source=source,results=rows,component_execution_completed=True,source_load=executor.proof,warmup=executor.warmup,actual_external_requests=0,native_actions_dispatched=False,weights_changed=False,original_actual_native_action_exact=True,source_is_development_evaluation=True,do_not_train=True,prototype_qualified=False,production_adopted=False,code_sha256=sha256(Path(__file__)))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
