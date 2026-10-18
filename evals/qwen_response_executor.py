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

    def __init__(self,assets,device='cuda:1',prompt_profile='contract-examples',attention_backend='math',pending_speech_protocol='array'):
        if prompt_profile not in ('base','contract-examples'):raise ValueError('Unknown local controller prompt profile')
        if attention_backend not in ('math','flash'):raise ValueError('Unknown explicit attention backend')
        if pending_speech_protocol not in ('array','explicit-actions'):raise ValueError('Unknown pending speech protocol')
        self.pending_speech_protocol=pending_speech_protocol
        self.attention_backend=attention_backend
        self.model,self.tokenizer,self.proof=load_model(assets,device=device)
        self.generator=Generator(self.model,self.tokenizer);self.model_id=self.proof['repo']
        self.prompt_profile=prompt_profile;self.lock=threading.RLock();self.history={};self.counter=0
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with torch.inference_mode(),torch.cuda.device(self.model.device),sdpa_kernel(SDPBackend.MATH if self.attention_backend=='math' else SDPBackend.FLASH_ATTENTION):
            self.warmup=self.generator.request([dict(role='user',content='Say ready.')],[])

    def __call__(self,payload):
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with self.lock,torch.inference_mode(),torch.cuda.device(self.model.device),sdpa_kernel(SDPBackend.MATH if self.attention_backend=='math' else SDPBackend.FLASH_ATTENTION):
            return self._request(payload)

    def parse_generated_action(self,text,definitions):return parse_action(text)
    def tool_history_arguments(self,arguments):return json.dumps(arguments)

    def model_functions(self,definitions,state):
        if self.pending_speech_protocol!='explicit-actions' or state['dispatch_protocol']!='state-aware':return definitions
        import copy
        result=copy.deepcopy(definitions)
        reply=next(d for d in result if d['name']=='respond_to_user')
        schema=reply['parameters'];schema['properties'].pop('consume_pending_speech_ids');schema['required'].remove('consume_pending_speech_ids')
        reply['description']+=' This reply preserves owed ordinary speech. Use for a lookup-cancellation confirmation alone.'
        if state['pending_speech']:
            resolve=copy.deepcopy(reply);resolve['name']='resolve_pending_speech'
            resolve['description']='Reply AND explicitly finish, withdraw or supersede identified owed ordinary answers. The reply must answer that request, or the latest user must withdraw/supersede it. Cancelling a private lookup alone NEVER resolves unrelated ordinary speech.'
            resolve['parameters']['properties']['speech_ids']=dict(type='array',minItems=1,items=dict(type='string',enum=[p['id'] for p in state['pending_speech']]))
            resolve['parameters']['required'].append('speech_ids');result.append(resolve)
            if state.get('domain')=='reasoning-development' and any(d['name']=='withdraw_speech' for d in definitions):
                resolve['description']='REPLACE the identified older answer with a NEW answer to the complete LATEST user request. Use for a shorter version, first step or summary. It retires the old version and speaks the revised answer. This is not silent withdrawal.'
                resolve['parameters']['properties']['content']['description']='New spoken answer matching the latest requested revision. If a shorter answer is requested, write that shorter answer; do not copy the full queued explanation.'
        if getattr(self,'compact_replies',False):
            aliases=[]
            for route in ('answer','clarify'):
                alias=copy.deepcopy(reply);alias['name']=route
                alias['parameters']['properties'].pop('route');alias['parameters']['required'].remove('route')
                alias['description']=('Answer ordinary knowledge or confirm an actual operation.' if route=='answer' else 'Ask for information missing from the latest request.')+' Preserve all owed ordinary speech.'
                aliases.append(alias)
            position=result.index(reply);result[position:position+1]=aliases
        return result

    def canonical_action(self,action,state):
        if getattr(self,'compact_replies',False) and action['name'] in ('answer','clarify'):
            action=dict(name='respond_to_user',arguments=dict(action['arguments'],route=action['name']))
        if self.pending_speech_protocol!='explicit-actions' or state['dispatch_protocol']!='state-aware':return action
        if action['name']=='respond_to_user':return dict(name=action['name'],arguments=dict(action['arguments'],consume_pending_speech_ids=[]))
        if action['name']=='resolve_pending_speech':
            arguments=dict(action['arguments']);identities=arguments.pop('speech_ids')
            return dict(name='respond_to_user',arguments=dict(arguments,consume_pending_speech_ids=identities))
        return action

    def _request(self,payload):
        if payload['model']!=self.model_id:raise RuntimeError('Local controller cannot substitute a different model')
        state=payload['local_worker_state']
        if state['dispatch_protocol'] not in ('required-action','state-aware'):raise RuntimeError('Unsupported local worker protocol')
        compact=getattr(self,'compact_replies',False)
        if compact and (self.pending_speech_protocol!='explicit-actions' or state['dispatch_protocol']!='state-aware'):raise ValueError('Compact replies require explicit state-aware speech actions')
        speech_choices='answer for an ordinary answer or clarify for missing information' if compact else 'respond_to_user for an ordinary answer or clarification'
        speech_answer='answer' if compact else 'respond_to_user'
        speech_usage='Use answer for an ordinary reply or cancellation confirmation, and clarify for missing information; this preserves all owed speech. ' if compact else 'Use respond_to_user for an ordinary reply, clarification, or confirmation of private lookup cancellation; this preserves all owed speech. '
        instructions=(
            'You control a voice assistant. Address ONLY the latest USER request in its chronological transcript. '
            'Earlier user requests and active tasks are context, not repeated commands. Never repeat an older lookup just because its result has not been spoken. '
            f'Choose exactly one provided function, including {speech_choices}. '
            'Return only the function call, never a preliminary plan or prose outside it. Answer simple general knowledge from your parameters. '
            'Private fields require an actual lookup; never infer a project from the available names. Ask for missing project or field identification. '
            'Honor negation and corrections. Use replace_lookup when an active lookup is corrected to a different user-identified project or field. '
            'Cancel only identified existing lookup IDs, all related duplicates but never current or unrelated work. '
            'Report completed operations only from their actual execution receipts, never from assistant speech. Keep user-facing content within35 words. '
            'Available records contain field NAMES only; their values have not been provided. ')
        domain=state.get('domain','records')
        if domain not in ('records','reasoning-development'):raise ValueError('Unknown controller task domain')
        if domain=='reasoning-development':
            instructions=(
                'You control a voice assistant. Address only the actual latest USER request. Earlier requests are context, never repeated commands. '
                'Choose exactly one provided function; return only its call. Answer simple knowledge and short arithmetic directly. '
                'Use delegate_reasoning for difficult multi-step calculation, combinatorial optimization or exhaustive counting. '
                'It executes a real stronger-model request with the original latest user question unchanged. Never invent its answer. '
                'If the user prohibits external tools or another model, do not delegate; answer locally or honestly state uncertainty. '
                'Already running tasks are real background work; do not repeat them when the user asks an unrelated ordinary question. '
                'Use cancel_reasoning when the user cancels an identified active reasoning result, preserving unrelated work. It stops result delivery, not already running remote computation; never claim that computation was aborted. '
                'The ACTUAL client state is authoritative about existing tasks. An empty tasks object means no earlier task is active. '
                'If there is no matching active task, explain that there is nothing to cancel or ask which task; never say cancellation happened. '
                'An ordinary answer does not execute an operation. Confirm a cancellation only through the actual cancel_reasoning action or its execution receipt. '
                'Keep speech natural and concise, at most35 words. Never narrate internal IDs, task state or scheduling metadata. ')
            if any(t['name']=='withdraw_speech' for t in payload.get('tools',[])):
                instructions+=('Use withdraw_speech for an explicit request only to stop talking: retire the identified pending speech without another spoken reply. '
                    'Do not answer or repeat the withdrawn explanation. Stopping speech does not cancel unrelated background work. '
                    'Read the WHOLE latest request: stop followed by a request for a first step, summary or new answer requires that answer, never silence. '
                    'If the user also asks a new question or revised explanation, answer it through '
                    +('resolve_pending_speech' if self.pending_speech_protocol=='explicit-actions' else 'respond_to_user with consume_pending_speech_ids')+' instead. ')
        if state.get('resumable_speech') is not None:
            resume_usage='choose resume_speech with no arguments; the application binds the sole retained generator shown in the actual state' if state.get('resume_action_protocol') else 'choose resume_speech with its exact speech_id and event_id'
            instructions+=('The actual state identifies a PAUSED speech generator retained by the client. '
                f'For a request to continue that explanation or a mere acknowledgment, {resume_usage}. '
                'This resumes the remaining audio without a new introduction, paraphrase or duplicate prefix. It is an output action, not completion of a task. '
                'For stop, a requested revision, another question or a cancellation, use the corresponding provided action instead. ')
        if any(t['name']=='cancel_reasoning_and_resume_speech' for t in payload.get('tools',[])):
            instructions+=('If the user cancels an active reasoning result AND wants the current explanation to continue, choose cancel_reasoning_and_resume_speech. '
                'Plain resume_speech would not execute the requested cancellation. Do not generate a cancellation announcement or restart the explanation; the combined action resumes the retained audio after its actual cancellation receipt. ')
        if state['dispatch_protocol']=='state-aware' and self.pending_speech_protocol=='array':
            instructions+=(
                'Pending speech lists ordinary answers owed but not fully played; an unknown prefix may have been heard. '
                'The client resumes each unless its ID is listed in consume_pending_speech_ids. Include an ID only when this reply fulfills '
                'that exact ordinary request or the latest user explicitly withdraws or supersedes it. Never both answer it and leave it queued. '
                'A latest user instruction to stop and replace an explanation with a shorter version supersedes the older longer answer; '
                'answer the revised request and consume its old pending speech ID so the old version does not resume. '
                'Cancellation of a lookup alone does not cancel an unrelated ordinary answer. Pending speech is not a private receipt or user project identification. ')
        if state['dispatch_protocol']=='state-aware' and self.pending_speech_protocol=='explicit-actions':
            instructions+=(
                'Pending speech contains ordinary answers owed but not fully played; an unknown prefix may have been heard. '
                f'{speech_usage}'
                'Use resolve_pending_speech with speech_ids ONLY if this reply itself answers an owed ordinary request, or the latest user explicitly withdraws or supersedes it. '
                'A user asking to stop and replace an explanation with a shorter version supersedes the older long answer: give the revised answer through resolve_pending_speech. '
                f'Cancelling a private lookup alone never resolves an unrelated ordinary answer; use {speech_answer} after that receipt and leave the client to resume it. '
                'These are output actions only, not execution receipts. Never both answer an owed request now and leave it queued. ')
            if domain=='reasoning-development' and any(t['name']=='withdraw_speech' for t in payload.get('tools',[])):
                instructions+=(
                    'Speech-control illustrations, NOT actual tasks or receipts: '
                    "If the user says 'Enough of the long version; give its main point instead', choose resolve_pending_speech with a NEW shorter answer and the old speech ID. Silent withdrawal would leave the new request unanswered. "
                    "If the user says 'End the calculation and carry on explaining', choose cancel_reasoning on the calculation with consume_pending_speech_ids=[]. A cancellation confirmation or promise to continue is not the explanation, so it cannot consume that explanation's ID. ")
        if self.prompt_profile=='contract-examples' and domain=='records':
            examples=CONTRACT_EXAMPLES
            if compact:examples=examples.replace('Choose respond_to_user\nwith route clarify and content','Choose clarify\nwith content')
            instructions+=examples
        if state['dispatch_protocol']=='state-aware' and self.pending_speech_protocol=='array':
            instructions+=(
                "\nSpeech-state examples, NOT actual tasks or receipts: "
                "An active lookup ID is lookup-demo and an owed ordinary explanation ID is speech-demo. "
                "The user says only 'Cancel the lookup.' Call cancel_lookup on lookup-demo. After its receipt, "
                "a reply that only says the lookup is cancelled MUST set consume_pending_speech_ids to []. "
                "Cancelling lookup-demo does not fulfil speech-demo. Empty [] is the usual correct value. "
                "If the user says 'Cancel the lookup but finish the explanation', either leave [] so the client resumes it, "
                "or include the actual explanation in this reply AND consume ['speech-demo']; never consume it with only a cancellation confirmation. "
                "If the user explicitly says 'Do not finish the explanation', consuming ['speech-demo'] withdraws it. ")
        compact=dict(state,available=dict(state['available'],tasks={k:{name:value for name,value in task.items() if name!='context'} for k,task in state['available']['tasks'].items()}))
        instructions+='\nACTUAL client state (chronological transcript supplied separately): '+json.dumps(compact)
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
                if event.get('control_source')=='actual-tool-execution':
                    if event.get('role')=='assistant':
                        calls=event.get('tool_calls')
                        if not isinstance(calls,list) or len(calls)!=1:raise RuntimeError('Control history requires one actual call')
                        call=calls[0];function=call['function']
                        if call['type']!='function' or function['name'] not in ('lookup_record','cancel_lookup','replace_lookup') or not isinstance(function['arguments'],dict):
                            raise RuntimeError('Invalid actual tool context')
                        messages.append(dict(role='assistant',content=None,tool_calls=[dict(type='function',id=call['id'],
                            function=dict(name=function['name'],arguments=self.tool_history_arguments(function['arguments'])))]))
                    elif event.get('role')=='tool':
                        if (not messages or messages[-1]['role']!='assistant' or len(messages[-1].get('tool_calls',[]))!=1
                                or messages[-1]['tool_calls'][0]['id']!=event.get('tool_call_id') or not isinstance(event.get('text'),str)):
                            raise RuntimeError('Actual tool notice lacks its matching chronological call')
                        messages.append(dict(role='tool',tool_call_id=event['tool_call_id'],content=event['text']))
                    else:raise RuntimeError('Invalid control event role')
                else:
                    if event.get('role') not in ('user','assistant') or not isinstance(event.get('text'),str):
                        raise RuntimeError('Unsupported chronological event; no transcript fallback is allowed')
                    messages.append(dict(role=event['role'],content=event['text']))
        if not prior:
            latest=next((event['text'] for event in reversed(context) if event['role']=='user'),None)
            if latest is None:raise RuntimeError('No actual latest user request')
            messages[0]['content']+='\nACTUAL latest USER request to address now: '+json.dumps(latest)
        if reply_only:
            schema=payload['text']['format']['schema']
            definitions=[dict(type='function',name='respond_to_user',
                description='Return the final user-facing response to the actual tool receipt. This is output formatting only, without external side effects.',parameters=schema)]
            messages.append(dict(role='user',content='Report the actual tool receipt using exactly one provided reply-format function. Return exactly one function call, never plain prose. Include every required field and no fields outside the selected function schema. Do not execute another external operation.'))
        else:
            if payload.get('tool_choice')!='required':raise RuntimeError('Local executor requires the explicit model-selected action protocol')
            definitions=sorted(payload['tools'],key=lambda f:f['name']!='respond_to_user')
        model_definitions=self.model_functions(definitions,state)
        if getattr(self,'spoken_output_contract',False):
            import copy
            model_definitions=copy.deepcopy(model_definitions)
            for definition in model_definitions:
                content=definition['parameters']['properties'].get('content')
                if content is not None:
                    content['description']='Spoken response in natural sentences. Express arithmetic in words, without written equations or Markdown.'
                    if reply_only:definition['description']+=' Report only this actual receipt; do not narrate other tasks or internal withheld-value notices.'
        tools=[dict(type='function',function={k:v for k,v in f.items() if k not in ('type','strict')}) for f in model_definitions]
        generation=self.generator.request(messages,tools,assistant_prefill='<tool_call>\n')
        try:
            native_action=self.parse_generated_action(generation['cleaned_text'],model_definitions)
            native_definition=next((d for d in model_definitions if d['name']==native_action['name']),None)
            if native_definition is None:raise ValueError('Model selected an unavailable function')
            validate(native_action['arguments'],native_definition['parameters'])
            action=self.canonical_action(native_action,state)
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
                function=dict(name=action['name'],arguments=self.tool_history_arguments(action['arguments'])))]))
        self.history[identity]=messages
        return dict(id=identity,status='completed',model=self.model_id,output=output,
            usage=dict(input_tokens=generation['input_tokens'],output_tokens=generation['generated_tokens'],
                total_tokens=generation['input_tokens']+generation['generated_tokens']),
            local_inference=dict(**generation,provider=self.provider_name,prompt_profile=self.prompt_profile,
                response_is_internal_normalized_transport=True,reply_function_is_format_only=reply_only,attention_backend=self.attention_backend,
                compact_reply_action_protocol=compact,pending_speech_protocol=self.pending_speech_protocol,native_model_action=native_action,canonical_worker_action=action))


def main():
    import argparse
    from pathlib import Path
    from evals.qwen_controller_probe import fixtures
    from evals.recovery_worker import RecordWorld,run_worker
    from evals.moshi_model_download import sha256
    ap=argparse.ArgumentParser(description='Actual typed worker-contract proof using only the local Qwen model')
    ap.add_argument('--assets',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--attention-backend',choices=('math','flash'),default='math')
    ap.add_argument('--pending-speech-protocol',choices=('array','explicit-actions'),default='array')
    ap.add_argument('--kernel-assets',type=Path)
    ap.add_argument('--spoken-output-contract',action='store_true')
    ap.add_argument('--compact-replies',action='store_true')
    ap.add_argument('--decode-backend',choices=('eager','static-eager','static-compiled'),default='eager')
    ap.add_argument('--controller-transport',choices=('in-process','subprocess'),default='in-process')
    ap.add_argument('--provider',choices=('qwen4b','qwen35'),default='qwen4b')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    if args.decode_backend!='eager' and (args.provider!='qwen35' or args.controller_transport!='in-process'):
        raise ValueError('Static decode experiment requires the isolated newer9B in-process component')
    if (args.kernel_assets is not None or args.spoken_output_contract or args.compact_replies) and args.provider!='qwen35':
        raise ValueError('Optional kernels and spoken contract require the newer9B controller')
    if args.controller_transport=='subprocess':
        if args.provider!='qwen35' or args.attention_backend!='math':raise ValueError('Resident proof supports the explicit newer model/math backend only')
        from evals.qwen35_rpc import ResidentQwen35
        executor=ResidentQwen35(args.assets,args.out,device='cuda:0',pending_speech_protocol=args.pending_speech_protocol,kernel_assets=args.kernel_assets,spoken_output_contract=args.spoken_output_contract,compact_replies=args.compact_replies)
    else:
        executor_class=QwenResponseExecutor
        if args.provider=='qwen35':
            from evals.qwen35_controller import Qwen35ResponseExecutor
            executor_class=Qwen35ResponseExecutor
        extra=dict(kernel_assets=args.kernel_assets,spoken_output_contract=args.spoken_output_contract,compact_replies=args.compact_replies) if args.provider=='qwen35' else {}
        executor=executor_class(args.assets,device='cuda:0',attention_backend=args.attention_backend,pending_speech_protocol=args.pending_speech_protocol,**extra)
    if args.decode_backend!='eager':
        from transformers import CompileConfig
        executor.model.generation_config.cache_implementation='static'
        executor.model.generation_config.disable_compile=args.decode_backend=='static-eager'
        if args.decode_backend=='static-compiled':
            executor.model.generation_config.compile_config=CompileConfig(fullgraph=True,mode='reduce-overhead')
    results=[]
    rows=fixtures()+[
        dict(id='ordinary-while-pending',records={'Oak Central':{'meeting room':'room seven'}},pending='Oak Central',kind='answer',
             history=[dict(role='user',text='Please check the meeting room for Project Oak Central.')],
             user='While you check, what does evaporation mean?'),
        dict(id='superseded-ordinary-format',records={},kind='answer',
             history=[dict(role='user',text='Explain in four sentences why seasons change.')],
             user='Stop. Give me just the main reason in one short sentence.',
             pending_speech=[dict(id='old-explanation',request_text='Explain in four sentences why seasons change.',
                text="Seasons change because Earth is tilted. Sunlight strikes different latitudes at different angles. Daylight lengths change during the year. The hemispheres have opposite seasons.")]),
    ]
    try:
        for row in rows:
            world=RecordWorld(row['records'])
            if row.get('pending'):world.read('pending-oak',row['pending'],'meeting room')
            revision=world.reserve('current')
            pending=row.get('pending_speech',[])
            if row['id']=='cancel':
                pending=[dict(id='ordinary',request_text='How many sides does a triangle have?',text='A triangle has three sides.')]
            try:
                result=run_worker(row.get('history',[])+[dict(role='user',text=row['user'])],world,'current',model=executor.model_id,
                    expected_revision=revision,dispatch_protocol='state-aware',pending_speech=pending,response_executor=executor)
                passed=result['route']==row['kind']
                if row['id'] in ('cancel','ordinary-while-pending'):passed=passed and result['consume_pending_speech_ids']==[]
                if row['id']=='ordinary-while-pending':passed=passed and not result['receipts']
                if row['id']=='superseded-ordinary-format':passed=passed and result['consume_pending_speech_ids']==['old-explanation']
                if row.get('replacement'):
                    passed=passed and world.tasks['pending-oak']['cancelled'] and world.tasks['current'].get('project')=='Pine Central'
                record=dict(id=row['id'],execution_completed=True,action_contract_passed=passed,result=result,world_tasks=world.tasks)
            except Exception as error:
                record=dict(id=row['id'],execution_completed=False,action_contract_passed=False,error_type=type(error).__name__,
                    message=str(error),failure=getattr(error,'result',None),world_tasks=world.tasks)
            (args.out/(row['id']+'.json')).write_text(json.dumps(record,indent=2)+'\n');results.append(record)
    finally:
        if hasattr(executor,'close'):executor.close()
    decode_proof=dict(requested=args.decode_backend,compiled_call_created=False,static_cache_created=False)
    if args.decode_backend!='eager':
        cache=getattr(executor.model,'_cache',None)
        decode_proof.update(compiled_call_created=callable(getattr(executor.model,'_compiled_call',None)),
            static_cache_created=type(cache).__name__=='StaticCache',cache_type=type(cache).__name__,
            cache_compileable=bool(cache is not None and cache.is_compileable),
            fullgraph_requested=args.decode_backend=='static-compiled',fallback_authorized=False)
    report=dict(schema='dl-local-qwen-worker-contract-v1',results=results,load=executor.proof,warmup=executor.warmup,
        optional_kernel_assets_sha256=sha256(args.kernel_assets/'report.json') if args.kernel_assets else None,
        controller_transport=args.controller_transport,transport_info=getattr(executor,'transport_info',None),
        decode_backend=decode_proof,
        compact_replies=args.compact_replies,spoken_output_contract=args.spoken_output_contract,provider=executor.provider_name,pending_speech_protocol=args.pending_speech_protocol,attention_backend=args.attention_backend,remote_api_used=False,input_is_typed=True,optimizer_steps=0,development_cases_reused=True,
        speech_generation_executed=False,duplex_system_qualified=False,code_sha256=sha256(Path(__file__)))
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
