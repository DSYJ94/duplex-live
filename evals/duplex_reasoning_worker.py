"""Opt-in development worker for actual stronger-model reasoning.

Local controller chooses a direct reply or a real external reasoning request.
The original latest user request is bound by the application; there are no
arbitrary actions, private-record reads or invented tool results. Cancellation
and replacement are intentionally not offered by this first component trial.
"""
import json
import os
import threading
import time
from evals.recovery_worker import RecordWorld,WorkerContractError,WorkerSuperseded,tool


def ask_reasoner(question):
    import requests
    before=time.monotonic()
    response=requests.post('https://api.openai.com/v1/responses',headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']},
        json=dict(model='gpt-6-astra',store=False,max_output_tokens=4096,
            instructions='Solve the actual user problem carefully. Give a concise spoken answer, at most35 words. Use natural words for numbers and arithmetic. Do not include Markdown, written formulas, plans, or internal metadata.',
            input=[dict(role='user',content=question)]),timeout=180)
    if response.status_code!=200:raise RuntimeError('Explicit stronger reasoning backend HTTP '+str(response.status_code))
    data=response.json()
    if data.get('status')!='completed':raise RuntimeError('Reasoning backend did not complete within its explicit token budget')
    content=''.join(p['text'] for m in data.get('output',[]) if m['type']=='message' for p in m['content'] if p['type']=='output_text').strip()
    if not content:raise RuntimeError('Reasoning backend returned no answer')
    return dict(content=content,response_id=data['id'],model=data['model'],usage=data.get('usage'),seconds=time.monotonic()-before,
        request_started_monotonic=before,request_finished_monotonic=time.monotonic(),remote_api_used=True)


class ReasoningWorld(RecordWorld):
    def __init__(self,records,reasoner=ask_reasoner):
        if records:raise ValueError('Reasoning component does not accept private records')
        super().__init__({});self.reasoner=reasoner;self.request_count=0
    def reason(self,task_id,revision,question):
        with self.lock:
            state=self.tasks[task_id]
            if state['revision']!=revision or state['cancelled'] or state['closed']:raise WorkerSuperseded('Reasoning task no longer active')
            if self.request_count>=2:raise RuntimeError('Two-request speech component limit reached')
            self.request_count+=1;state.update(operation='delegate_reasoning',phase='running',question=question)
        # Do not hold the world lock or local controller lock across the API call.
        result=self.reasoner(question)
        receipt=dict(task_id=task_id,revision=revision,tool='delegate_reasoning',question=question,read_only=True,executed=True,**result)
        with self.lock:self.receipts.append(receipt)
        return receipt


def reasoning_receipts(receipts):
    if any(r.get('tool')!='delegate_reasoning' or not r.get('executed') or not r.get('remote_api_used') or not r.get('content') for r in receipts):
        raise ValueError('Reasoning result lacks its actual execution receipt')
    return receipts


def run_reasoning_worker(context,world,task_id,model,expected_revision,dispatch_protocol,pending_speech,response_executor,
                         scope_read_tools=False,execution_observer=None):
    if dispatch_protocol!='state-aware' or response_executor is None or scope_read_tools or execution_observer is not None:
        raise ValueError('Reasoning trial requires local state-aware control and no record-specific projection')
    latest=next((e['text'] for e in reversed(context) if e.get('role')=='user'),None)
    if not latest:raise ValueError('No actual latest user question')
    reply=tool('respond_to_user','Answer easy knowledge directly or ask for clarification. Content is read aloud.',dict(
        route=dict(type='string',enum=['answer','clarify']),content=dict(type='string'),consume_pending_speech_ids=dict(type='array',items=dict(type='string'))))
    delegate=tool('delegate_reasoning','Ask the stronger reasoning model to solve a difficult multi-step problem, counting or optimization. The application sends the complete original latest user question unchanged; this call returns its actual answer.',{})
    with world.lock:
        available=dict(records={},tasks={k:{name:value for name,value in v.items() if name!='context'} for k,v in world.tasks.items()
            if k!=task_id and not v['closed'] and not v['cancelled']})
    payload=dict(model=model,input=json.dumps(context),tools=[reply,delegate],tool_choice='required',
        local_worker_state=dict(available=available,pending_speech=pending_speech,dispatch_protocol=dispatch_protocol,domain='reasoning-development'))
    before=time.monotonic();response=response_executor(payload)
    calls=[c for c in response.get('output',[]) if c.get('type')=='function_call']
    if response.get('status')!='completed' or len(calls)!=1:raise WorkerContractError('Reasoning controller did not choose exactly one action',dict(response=response))
    call=calls[0];arguments=json.loads(call['arguments'])
    event=dict(response_id=response['id'],model=response['model'],provider=response_executor.provider_name,usage=response.get('usage'),
        seconds=time.monotonic()-before,local_inference=response.get('local_inference'),output=response['output'])
    if call['name']=='respond_to_user':
        route=arguments['route'];content=arguments['content'];consumed=arguments['consume_pending_speech_ids'];receipts=[]
    elif call['name']=='delegate_reasoning':
        if arguments:raise WorkerContractError('Bound reasoning action cannot rewrite the user question',dict(response=response))
        try:receipt=world.reason(task_id,expected_revision,latest)
        except Exception as error:raise WorkerContractError('Actual reasoning execution failed: '+str(error),dict(controller_event=event,error_type=type(error).__name__)) from error
        route='delegate';content=receipt['content'];receipts=[receipt];consumed=[]
    else:raise WorkerContractError('Unavailable reasoning action',dict(response=response))
    if not isinstance(content,str) or not content.strip():raise WorkerContractError('Reasoning worker produced empty speech',dict(response=response))
    if not isinstance(consumed,list) or len(set(consumed))!=len(consumed) or not set(consumed)<={p['id'] for p in pending_speech}:raise WorkerContractError('Unknown owed speech identity',dict(response=response))
    return dict(route=route,content=content,consume_pending_speech_ids=consumed,receipts=receipts,events=[event],task_id=task_id,
        speech_model_routing=False,final_response_route=route,dispatch_protocol=dispatch_protocol,response_provider=response_executor.provider_name,
        actual_reasoning_backend=receipts[0]['model'] if receipts else None,agent_actions_executed=False,cancellation_supported=False)
