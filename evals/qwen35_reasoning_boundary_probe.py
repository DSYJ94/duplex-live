"""Typed development test of local routing to an actual stronger reasoner.

No speech, agent actions or training. Exact CPU oracles are withheld from every
model prompt. Counterfactual direct local answers measure whether delegation
adds correctness. The remote API is explicit, bounded and never a fallback.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import os
from pathlib import Path
import re
import time
from evals.moshi_model_download import sha256
from evals.recovery_worker import tool


def cases():
    rows=[dict(id='sides',question='How many sides does a triangle have?',expected=3,complex=False),
          dict(id='pencils',question='Three boxes each contain four pencils. You give away two pencils. How many remain?',expected=10,complex=False),
          dict(id='addition',question='What is nine plus eight?',expected=17,complex=False)]
    weights=[4,7,2,9,5,3,8,6,11,1,10,12];values=[11,15,6,21,12,8,18,14,25,4,22,26]
    for capacity in (23,31):
        best=max(sum(v for v,b in zip(values,bits) if b) for bits in itertools.product((0,1),repeat=len(weights)) if sum(w for w,b in zip(weights,bits) if b)<=capacity)
        rows.append(dict(id='packing-'+str(capacity),question='Each item can be taken at most once. Item weights are '+json.dumps(weights)+', and corresponding values are '+json.dumps(values)+'. With total weight at most '+str(capacity)+', what is the maximum total value? Return the value, not a chosen item list.',expected=best,complex=True))
    matrix=[[17,9,23,11,28,14],[12,25,8,19,16,21],[24,13,18,7,22,15],[10,20,14,26,9,17],[21,8,27,16,13,11],[15,22,10,18,12,24]]
    best=min(sum(matrix[i][j] for i,j in enumerate(p)) for p in itertools.permutations(range(6)))
    rows.append(dict(id='assignment',question='Six workers must each take a different one of six jobs, and every job must be assigned once. Rows are workers and columns are jobs. The cost matrix is '+json.dumps(matrix)+'. What is the minimum possible total cost?',expected=best,complex=True))
    numbers=[2,3,5,7,11,13,17,19,23,29,31,37];target=50
    count=sum(sum(v for v,b in zip(numbers,bits) if b)==target for bits in itertools.product((0,1),repeat=len(numbers)))
    rows.append(dict(id='subset-count',question='From '+json.dumps(numbers)+', how many distinct subsets sum to fifty? Each listed number can be used at most once; order does not matter. Return the number of subsets.',expected=count,complex=True))
    return rows


def integer(text):
    match=re.fullmatch(r'\s*(-?\d+)\s*',text)
    if not match:raise ValueError('Answer is not exactly an integer; no text repair')
    return int(match.group(1))


def remote(question,model,origin):
    import requests
    started=time.monotonic()
    response=requests.post('https://api.openai.com/v1/responses',headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']},
        json=dict(model=model,store=False,max_output_tokens=4096,
                  instructions='Solve the user problem carefully. Return only the final integer, without explanation or formatting.',
                  input=[dict(role='user',content=question)]),timeout=180)
    if response.status_code!=200:raise RuntimeError('Explicit reasoning backend HTTP '+str(response.status_code))
    data=response.json()
    if data.get('status')!='completed':raise RuntimeError('Explicit reasoning backend did not complete within its token budget')
    text=''.join(p['text'] for m in data.get('output',[]) if m['type']=='message' for p in m['content'] if p['type']=='output_text')
    answer=integer(text)
    return dict(answer=answer,content=text,model=data['model'],response_id=data['id'],usage=data.get('usage'),seconds=time.monotonic()-started,request_started_s=started-origin,request_completed_s=time.monotonic()-origin,executed=True,remote_api_used=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--kernel-assets',type=Path,required=True)
    ap.add_argument('--reasoner',default='gpt-6-astra');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if not os.environ.get('OPENAI_API_KEY'):raise RuntimeError('Explicit reasoning pilot requires the existing API key')
    args.out.mkdir(parents=True,exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    executor=Qwen35ResponseExecutor(args.assets,kernel_assets=args.kernel_assets,spoken_output_contract=True)
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    definitions=[dict(type='function',function={k:v for k,v in t.items() if k!='type'}) for t in (
        tool('respond_to_user','Answer from your own knowledge; content must be the final integer only.',{'content':dict(type='string')}),
        tool('delegate_reasoning','Ask an available stronger reasoning model to solve a multi-step problem. It returns an actual answer asynchronously. Do not invent its result.',{'question':dict(type='string')}))]
    def local(question,allow_delegate=True):
        instructions=('Choose exactly one function. Answer simple facts and short arithmetic yourself. Delegate combinatorial optimization and exhaustive counting when reliable calculation exceeds your immediate ability. '
                      'Delegation is a real external model call, not a private-record lookup. If delegating, pass the complete original problem without adding an answer. '
                      'Return only the function call. An ordinary answer content is the final integer only.')
        tools=definitions if allow_delegate else definitions[:1]
        with torch.inference_mode(),torch.cuda.device(executor.model.device),sdpa_kernel(SDPBackend.MATH):
            stage=executor.generator.request([dict(role='system',content=instructions),dict(role='user',content=question)],tools,assistant_prefill='<tool_call>\n')
        action=executor.parse_generated_action(stage['cleaned_text'],tools)
        return dict(action=action,inference=stage)
    rows=[];pool=ThreadPoolExecutor(max_workers=1);origin=time.monotonic();pending=[]
    try:
        for case in cases():
            row=dict(id=case['id'],question=case['question'],complex=case['complex'],expected_integer=case['expected'],oracle_supplied_to_models=False)
            try:
                if case['complex']:
                    try:
                        direct=local(case['question'],False);row['direct_local']=direct
                        row['direct_local_correct']=integer(direct['action']['arguments']['content'])==case['expected']
                    except Exception as error:row.update(direct_local_correct=False,direct_local_error=dict(type=type(error).__name__,message=str(error)))
                selected=local(case['question']);row['selection']=selected;action=selected['action']
                if action['name']=='respond_to_user':row.update(answer=integer(action['arguments']['content']),remote_api_used=False)
                elif action['name']=='delegate_reasoning':
                    # Exact question binding prevents omissions, hallucinated constraints,
                    # answer contamination or a different request from reaching the backend.
                    if action['arguments']['question']!=case['question']:raise ValueError('Delegated question differs from the original problem')
                    row['submitted_s']=time.monotonic()-origin;future=pool.submit(remote,case['question'],args.reasoner,origin);pending.append((row,future))
                    before=time.monotonic();foreground=local('What is two plus three?',False)
                    row['concurrent_foreground']=dict(result=foreground,correct=integer(foreground['action']['arguments']['content'])==5,seconds=time.monotonic()-before,
                        background_finished_at_foreground_completion=future.done(),completed_s=time.monotonic()-origin)
                else:raise ValueError('Unsupported action')
            except Exception as error:row['error']=dict(type=type(error).__name__,message=str(error))
            rows.append(row);(args.out/(case['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
        for row,future in pending:
            try:
                receipt=future.result();foreground=row.get('concurrent_foreground')
                if foreground:
                    end=foreground['completed_s'];begin=end-foreground['seconds']
                    foreground['overlaps_actual_remote_request']=begin<receipt['request_completed_s'] and end>receipt['request_started_s']
                row.update(reasoner_receipt=receipt,answer=receipt['answer'],remote_api_used=True,completed_s=time.monotonic()-origin)
            except Exception as error:row['error']=dict(type=type(error).__name__,message=str(error))
        for row in rows:
            row['correct']=not row.get('error') and row.get('answer')==row['expected_integer']
            (args.out/(row['id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    finally:pool.shutdown(wait=True,cancel_futures=True)
    report=dict(schema='dl-local-routing-actual-reasoner-typed-v1',results=rows,load=executor.proof,reasoner_requested=args.reasoner,
        routing_policy_instructions_explicit=True,development_cases=True,optimizer_steps=0,speech_executed=False,
        cancellation_tested=False,agent_actions_executed=False,duplex_qualified=False,code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
