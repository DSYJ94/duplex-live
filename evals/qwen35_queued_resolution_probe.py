"""One predeclared instruction/description probe using existing speech actions.

No new action, runtime dispatch, training or automatic user-keyword routing.
Reuses development payloads and explicitly cannot qualify general behavior.
"""
import argparse,copy,json
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.duplex_actual_controller import capture
from evals.qwen35_queued_delivery_probe import CASES

POLICY='''Speech accounting rule for generated-not-started ordinary answers:
A pending answer is owed speech, not an independent command to repeat later.
Choose resolve_pending_speech when your current reply fulfills the SAME pending request, even if the user only adds context already covered by that answer. Copying the prepared text is allowed when it remains fully correct; retire that speech ID so it is delivered once.
Choose resolve_pending_speech also when the latest request supersedes that answer with a different answer. Write the requested replacement and retire the superseded ID.
Choose answer without retiring that ID only for a genuinely separate question when the earlier answer is still wanted. Choose withdraw_speech for a stop with no replacement reply. Do not retire unrelated speech or cancel background work.
Illustrations, not actual conversation or receipts:
Prepared unsaid reply to 'Can you see my bedroom?' says 'I cannot see your bedroom.' User adds 'The one upstairs.' Resolve the old ID with one honest answer about the same inability, rather than enqueueing two answers.
Prepared unsaid instructions for arranging flowers; user says 'Skip those instructions and tell me the opposite of north.' Resolve the old ID with 'South.'
Prepared unsaid instructions for arranging flowers; user says 'Keep those instructions; first tell me the opposite of north.' Answer 'South.' while preserving the instructions.
These rules do not apply to an already started paused generator: use its existing resume protocol instead of repeating its full text.'''


class ContractGenerator:
    def __init__(self,original):self.original=original
    def request(self,messages,tools,assistant_prefill=''):
        messages=copy.deepcopy(messages);tools=copy.deepcopy(tools)
        messages[0]['content']+='\n'+POLICY
        for t in tools:
            f=t['function']
            if f['name']=='resolve_pending_speech':
                f['description']='Speak once to fulfill or supersede identified pending ordinary answers, and retire those IDs. An unspoken prepared answer may be copied if it still fully answers the same request. Otherwise write the requested replacement. Never retire unrelated speech.'
                f['parameters']['properties']['content']['description']='The single spoken answer that fulfills the same pending request or its latest replacement.'
        return self.original.request(messages,tools,assistant_prefill=assistant_prefill)


def assess(action,case,prepared):
    from data.synthetic.filter import norm_words
    args=action['arguments'];name=action['name'];cid=case['id']
    expected='withdraw_speech' if cid=='withdraw-prepared-reply' else 'answer' if cid=='preserve-prepared-new-question' else 'resolve_pending_speech'
    checks=dict(native_function=name==expected)
    if expected in ('withdraw_speech','resolve_pending_speech'):checks['exact_queued_identity']=args.get('speech_ids')==['turn-1']
    if cid in ('actual-additional-records-context','explicit-prepared-reply'):checks['prepared_answer_preserved']=norm_words(args.get('content',''))==norm_words(prepared)
    if case.get('number'):checks['requested_number_only']=norm_words(args.get('content',''))==norm_words(case['number'])
    return dict(passed=all(checks.values()),checks=checks)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','assets','kernel-assets','out'):ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    payload,source=capture(a.runs/'duplex-live-transport-7520301','turn-2')
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    executor=Qwen35ResponseExecutor(a.assets,device='cuda:0',kernel_assets=a.kernel_assets,pending_speech_protocol='explicit-actions',compact_replies=True);warm_long_prefill(executor)
    baseline=executor(copy.deepcopy(payload));b=baseline['local_inference']
    (a.out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    if b['input_tokens']!=source['original_input_tokens'] or b['native_model_action']['name']!='answer' or b['canonical_worker_action']['arguments']['consume_pending_speech_ids']!=[]:raise ValueError('Original duplication fault not reproduced')
    pending=payload['local_worker_state']['pending_speech']
    if len(pending)!=1 or payload['local_worker_state'].get('resumable_speech') is not None:raise ValueError('Requires actual sole prepared unspoken reply')
    executor.generator=ContractGenerator(executor.generator);rows=[]
    for case in CASES:
        p=copy.deepcopy(payload)
        if case['user'] is not None:
            context=json.loads(p['input']);context[-1]=dict(role='user',text=case['user']);p['input']=json.dumps(context)
        try:
            response=executor(p);action=response['local_inference']['native_model_action']
            row=dict(case=case,response=response,mechanical_contract=assess(action,case,pending[0]['text']))
        except Exception as error:row=dict(case=case,error_type=type(error).__name__,message=str(error),evidence=getattr(error,'result',None),mechanical_contract=dict(passed=False))
        rows.append(row);(a.out/(case['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(dict(case=case['id'],contract=row['mechanical_contract'])),flush=True)
    report=dict(schema='dl-existing-queued-resolution-proposal-v1',component_execution_completed=True,source=source,baseline=baseline,baseline_fault_reproduced=True,policy=POLICY,results=rows,source_load=executor.proof,all_mechanical_contracts_passed=all(r['mechanical_contract']['passed'] for r in rows),actual_external_requests=0,actions_dispatched=False,new_actions=0,do_not_train=True,weights_changed=False,optimizer_steps=0,production_adopted=False,native_duplex_qualified=False,code_sha256=sha256(Path(__file__)))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
