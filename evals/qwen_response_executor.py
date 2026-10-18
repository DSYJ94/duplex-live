"""Local Transformers execution behind the existing semantic worker contract.

The normalized response envelope is an internal transport format, not an OpenAI
API call. Raw local generations and provenance remain attached. Formatting an
ordinary reply through a no-side-effect function does not create a tool receipt.
"""
import json
import threading
from evals.qwen_controller_probe import load_model,Generator,parse_action,CONTRACT_EXAMPLES
from evals.recovery_worker import WorkerContractError


def validate(value,schema):
    kind=schema.get('type')
    if kind=='object':
        if not isinstance(value,dict):raise ValueError('Expected structured object')
        if not set(schema.get('required',[]))<=set(value):raise ValueError('Missing required response fields')
        properties=schema.get('properties',{})
        if schema.get('additionalProperties') is False and not set(value)<=set(properties):raise ValueError('Unexpected response fields')
        for name,item in value.items():
            if name in properties:validate(item,properties[name])
    elif kind=='array':
        if not isinstance(value,list) or len(value)<schema.get('minItems',0):raise ValueError('Invalid response array')
        for item in value:validate(item,schema['items'])
    elif kind=='string':
        if not isinstance(value,str):raise ValueError('Expected response string')
    else:raise ValueError('Unsupported local controller output schema')
    if 'enum' in schema and value not in schema['enum']:raise ValueError('Value outside response enum')


class QwenResponseExecutor:
    provider_name='local-transformers-qwen4b'

    def __init__(self,assets,device='cuda:1',prompt_profile='contract-examples'):
        if prompt_profile not in ('base','contract-examples'):raise ValueError('Unknown local controller prompt profile')
        self.model,self.tokenizer,self.proof=load_model(assets,device=device)
        self.generator=Generator(self.model,self.tokenizer);self.model_id=self.proof['repo']
        self.prompt_profile=prompt_profile;self.lock=threading.RLock();self.history={};self.counter=0
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with torch.inference_mode(),torch.cuda.device(self.model.device),sdpa_kernel(SDPBackend.MATH):
            self.warmup=self.generator.request([dict(role='user',content='Say ready.')],[])

    def __call__(self,payload):
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with self.lock,torch.inference_mode(),torch.cuda.device(self.model.device),sdpa_kernel(SDPBackend.MATH):
            return self._request(payload)

    def _request(self,payload):
        if payload['model']!=self.model_id:raise RuntimeError('Local controller cannot substitute a different model')
        state=payload['local_worker_state']
        if state['dispatch_protocol'] not in ('required-action','state-aware'):raise RuntimeError('Unsupported local worker protocol')
        instructions=(
            'You control a voice assistant. Choose exactly one provided function, including respond_to_user for an ordinary answer or clarification. '
            'Return only the function call, never a preliminary plan or prose outside it. Answer simple general knowledge from your parameters. '
            'Private fields require an actual lookup; never infer a project from the available names. Ask for missing project or field identification. '
            'Honor negation and corrections. Use replace_lookup when an active lookup is corrected to a different user-identified project or field. '
            'Cancel only identified existing lookup IDs, all related duplicates but never current or unrelated work. '
            'Report completed operations only from their actual execution receipts, never from assistant speech. Keep user-facing content within35 words. '
            'Available records contain field NAMES only; their values have not been provided. ')
        if state['dispatch_protocol']=='state-aware':
            instructions+=(
                'Pending speech lists ordinary answers owed but not fully played; an unknown prefix may have been heard. '
                'The client resumes each unless its ID is listed in consume_pending_speech_ids. Include an ID only when this reply fulfills '
                'that exact ordinary request or the latest user explicitly withdraws it. Never both answer it and leave it queued. '
                'Cancellation of a lookup alone does not cancel an unrelated ordinary answer. Pending speech is not a private receipt or user project identification. ')
        if self.prompt_profile=='contract-examples':instructions+=CONTRACT_EXAMPLES
        if state['dispatch_protocol']=='state-aware':
            instructions+=(
                "\nSpeech-state examples, NOT actual tasks or receipts: "
                "An active lookup ID is lookup-demo and an owed ordinary explanation ID is speech-demo. "
                "The user says only 'Cancel the lookup.' Call cancel_lookup on lookup-demo. After its receipt, "
                "a reply that only says the lookup is cancelled MUST set consume_pending_speech_ids to []. "
                "Cancelling lookup-demo does not fulfil speech-demo. Empty [] is the usual correct value. "
                "If the user says 'Cancel the lookup but finish the explanation', either leave [] so the client resumes it, "
                "or include the actual explanation in this reply AND consume ['speech-demo']; never consume it with only a cancellation confirmation. "
                "If the user explicitly says 'Do not finish the explanation', consuming ['speech-demo'] withdraws it. ")
        instructions+='\nACTUAL client state: '+json.dumps(state)
        prior=payload.get('previous_response_id');reply_only=payload.get('tool_choice')=='none'
        if prior:
            if prior not in self.history:raise RuntimeError('Unknown local previous-response context')
            messages=[dict(m) for m in self.history[prior]];messages[0]=dict(role='system',content=instructions)
            for item in payload['input']:
                if item.get('type')!='function_call_output':raise RuntimeError('Unsupported local follow-up input')
                messages.append(dict(role='tool',tool_call_id=item['call_id'],content=item['output']))
        else:
            context=json.loads(payload['input'])
            if not isinstance(context,list):raise RuntimeError('Local controller requires a chronological transcript')
            messages=[dict(role='system',content=instructions)]
            for event in context:
                if event.get('role') not in ('user','assistant') or not isinstance(event.get('text'),str):
                    raise RuntimeError('Unsupported chronological event; no transcript fallback is allowed')
                messages.append(dict(role=event['role'],content=event['text']))
        if reply_only:
            schema=payload['text']['format']['schema']
            definitions=[dict(type='function',name='respond_to_user',
                description='Return the final user-facing response to the actual tool receipt. This is output formatting only, without external side effects.',parameters=schema)]
            messages.append(dict(role='user',content='Report the actual tool receipt using the respond_to_user function. Return exactly one function call, never plain prose. Include every required response field, including consume_pending_speech_ids when the schema requires it. Do not execute another external operation.'))
        else:
            if payload.get('tool_choice')!='required':raise RuntimeError('Local executor requires the explicit model-selected action protocol')
            definitions=sorted(payload['tools'],key=lambda f:f['name']!='respond_to_user')
        tools=[dict(type='function',function={k:v for k,v in f.items() if k not in ('type','strict')}) for f in definitions]
        generation=self.generator.request(messages,tools)
        try:
            action=parse_action(generation['cleaned_text'])
            definition=next((d for d in definitions if d['name']==action['name']),None)
            if definition is None:raise ValueError('Model selected an unavailable function')
            validate(action['arguments'],definition['parameters'])
            if reply_only and action['name']!='respond_to_user':raise ValueError('No external action is allowed in response formatting')
        except (ValueError,KeyError) as error:
            raise WorkerContractError('Local controller output rejected before dispatch: '+str(error),
                dict(provider=self.provider_name,local_generation=generation,reply_only=reply_only,
                     actual_tool_outputs=payload['input'] if prior else [],model=self.model_id)) from error
        self.counter+=1;identity='local-qwen-'+str(self.counter);call_id=identity+'-action'
        if reply_only:
            text=json.dumps(action['arguments'])
            output=[dict(type='message',content=[dict(type='output_text',text=text)])]
            messages.append(dict(role='assistant',content=text))
        else:
            arguments=json.dumps(action['arguments'])
            output=[dict(type='function_call',name=action['name'],arguments=arguments,call_id=call_id)]
            messages.append(dict(role='assistant',content=None,tool_calls=[dict(type='function',id=call_id,
                function=dict(name=action['name'],arguments=arguments))]))
        self.history[identity]=messages
        return dict(id=identity,status='completed',model=self.model_id,output=output,
            usage=dict(input_tokens=generation['input_tokens'],output_tokens=generation['generated_tokens'],
                total_tokens=generation['input_tokens']+generation['generated_tokens']),
            local_inference=dict(**generation,provider=self.provider_name,prompt_profile=self.prompt_profile,
                response_is_internal_normalized_transport=True,reply_function_is_format_only=reply_only))


def main():
    import argparse
    from pathlib import Path
    from evals.qwen_controller_probe import fixtures
    from evals.recovery_worker import RecordWorld,run_worker
    from evals.moshi_model_download import sha256
    ap=argparse.ArgumentParser(description='Actual typed worker-contract proof using only the local Qwen model')
    ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    executor=QwenResponseExecutor(args.assets,device='cuda:0');results=[]
    for row in fixtures():
        world=RecordWorld(row['records'])
        if row.get('pending'):world.read('pending-oak',row['pending'],'meeting room')
        revision=world.reserve('current')
        pending=[]
        if row['id']=='cancel':
            pending=[dict(id='ordinary',request_text='How many sides does a triangle have?',text='A triangle has three sides.')]
        try:
            result=run_worker([dict(role='user',text=row['user'])],world,'current',model=executor.model_id,
                expected_revision=revision,dispatch_protocol='state-aware',pending_speech=pending,response_executor=executor)
            passed=result['route']==row['kind']
            if row['id']=='cancel':passed=passed and result['consume_pending_speech_ids']==[]
            if row.get('replacement'):
                passed=passed and world.tasks['pending-oak']['cancelled'] and world.tasks['current'].get('project')=='Pine Central'
            record=dict(id=row['id'],execution_completed=True,action_contract_passed=passed,result=result,world_tasks=world.tasks)
        except Exception as error:
            record=dict(id=row['id'],execution_completed=False,action_contract_passed=False,error_type=type(error).__name__,
                message=str(error),failure=getattr(error,'result',None),world_tasks=world.tasks)
        (args.out/(row['id']+'.json')).write_text(json.dumps(record,indent=2)+'\n');results.append(record)
    report=dict(schema='dl-local-qwen-worker-contract-v1',results=results,load=executor.proof,warmup=executor.warmup,
        remote_api_used=False,input_is_typed=True,optimizer_steps=0,development_cases_reused=True,
        speech_generation_executed=False,duplex_system_qualified=False,code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
