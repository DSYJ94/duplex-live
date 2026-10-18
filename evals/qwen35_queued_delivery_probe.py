"""Frozen-model proposal test for delivering one already prepared reply.

The existing client can only reply again, withdraw, or resolve owed speech when
no generator has started. This tests an explicit zero-word delivery action;
no media/runtime implementation, dispatch, API call or training is performed.
"""
import argparse,copy,json
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.duplex_actual_controller import capture

CASES=[
 dict(id='actual-additional-records-context',user=None,allowed=['continue_queued_speech','resolve_pending_speech']),
 dict(id='explicit-prepared-reply',user='Please go ahead with the answer you have already prepared.',allowed=['continue_queued_speech']),
 dict(id='withdraw-prepared-reply',user='Stop. I do not need any answer. Please say nothing more.',allowed=['withdraw_speech']),
 dict(id='replace-prepared-reply',user='Set that aside. What is six plus seven? Give just the number.',allowed=['resolve_pending_speech'],number='thirteen'),
 dict(id='preserve-prepared-new-question',user='I still want your previous reply, but first what is ten minus four? Give only the number.',allowed=['answer'],number='six'),
]


def proposed_payload(original,user=None):
    from evals.recovery_worker import tool
    p=copy.deepcopy(original);state=p['local_worker_state'];pending=state['pending_speech']
    if len(pending)!=1 or state.get('resumable_speech') is not None:raise ValueError('New action requires one actually queued, not currently playing, reply')
    state['queued_speech_delivery']=dict(speech_id=pending[0]['id'],delivery='generated-not-started',action_protocol='application-bound-single-queued-answer')
    p['tools'].append(tool('continue_queued_speech','Deliver the sole already prepared ordinary reply shown in queued_speech_delivery, once, WITHOUT generating any extra words. Use when the latest user adds context or acknowledgment that the prepared answer already covers, or asks to hear that prepared answer. This is speech delivery only, not a backend operation. Do not use for a new question, requested change, withdrawal, stop or cancellation. The application binds the exact queued reply; supply no arguments.',{}))
    if user is not None:
        context=json.loads(p['input']);context[-1]=dict(role='user',text=user);p['input']=json.dumps(context)
    return p


def assess(action,case):
    from data.synthetic.filter import norm_words
    checks=dict(native_function=action['name'] in case['allowed']);args=action['arguments']
    if action['name']=='continue_queued_speech':checks['bound_zero_arguments']=args=={}
    if action['name'] in ('withdraw_speech','resolve_pending_speech'):checks['exact_queued_identity']=args.get('speech_ids')==['turn-1']
    if case.get('number'):checks['requested_number_only']=norm_words(args.get('content',''))==norm_words(case['number'])
    if case['id']=='actual-additional-records-context' and action['name']=='resolve_pending_speech':checks['nonempty_reply_for_separate_semantic_review']=bool(args.get('content','').strip())
    return dict(passed=all(checks.values()),checks=checks,semantic_review_required=case['id']=='actual-additional-records-context' and action['name']=='resolve_pending_speech')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','assets','kernel-assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);payload,source=capture(a.runs/'duplex-live-transport-7520301','turn-2')
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    executor=Qwen35ResponseExecutor(a.assets,device='cuda:0',kernel_assets=a.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True);warm_long_prefill(executor)
    baseline=executor(copy.deepcopy(payload));r=baseline['local_inference']
    (a.out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    if r['input_tokens']!=source['original_input_tokens'] or r['native_model_action']['name']!='answer' or r['canonical_worker_action']['arguments']['consume_pending_speech_ids']!=[]:raise ValueError('Original queued-reply fault was not reproduced; comparison invalid')
    rows=[]
    for case in CASES:
        try:
            response=executor(proposed_payload(payload,case['user']));action=response['local_inference']['native_model_action'];quality=assess(action,case)
            row=dict(case=case,response=response,mechanical_contract=quality)
        except Exception as error:row=dict(case=case,error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None),mechanical_contract=dict(passed=False))
        rows.append(row);(a.out/(case['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(dict(case=case['id'],contract=row['mechanical_contract'])),flush=True)
    report=dict(schema='dl-queued-speech-delivery-proposals-v1',component_execution_completed=True,source=source,baseline=baseline,baseline_queued_reply_fault_reproduced=True,historical_baseline_native_output_exact=r['native_model_action']==source['original_native_action'],results=rows,source_load=executor.proof,all_mechanical_contracts_passed=all(r['mechanical_contract']['passed'] for r in rows),actual_external_requests=0,actions_dispatched=False,new_action_only_proposed_not_implemented=True,do_not_train=True,weights_changed=False,optimizer_steps=0,production_adopted=False,prototype_qualified=False,code_sha256=sha256(Path(__file__)))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
