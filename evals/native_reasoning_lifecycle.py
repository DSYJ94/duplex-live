"""Opt-in asynchronous lifecycle boundary for future native reasoning teachers.

The voice model must request work itself. This module never examines intent,
routes by a case label, calls an API, supplies an oracle, or generates speech.
Model-selected actions are committed against real current task identities before
their returned content can be sent. Raw provider/voice traces still need audit.
"""
import asyncio
import copy
import hashlib
import json
import math


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def transcript_snapshot(events,at):
    if not isinstance(events,list) or not events:raise ValueError('Actual arrived USER transcript required')
    previous=0.
    for event in events:
        if (set(event)!={'speaker','text','received_at_s'} or event['speaker']!='user'
                or not isinstance(event['text'],str) or not event['text']
                or type(event['received_at_s']) not in (int,float)
                or not math.isfinite(event['received_at_s']) or not previous<=event['received_at_s']<=at):
            raise ValueError('Only chronological actual arrived user words belong in a request snapshot')
        previous=event['received_at_s']
    return copy.deepcopy(events)


def function(name,properties,description):
    return dict(type='function',name=name,description=description,strict=True,
        parameters=dict(type='object',properties=properties,required=list(properties),additionalProperties=False))


class NativeReasoningLifecycle:
    def __init__(self,clock,minimum_delivery_delay_s,*,maximum_requests=6):
        if (type(minimum_delivery_delay_s) not in (int,float) or not math.isfinite(minimum_delivery_delay_s)
                or not 0<=minimum_delivery_delay_s<=30 or type(maximum_requests) is not int
                or not 1<=maximum_requests<=6):raise ValueError('Explicit finite source transport delay and request bound required')
        self.clock=clock;self.delay=minimum_delivery_delay_s;self.maximum=maximum_requests
        self.lock=asyncio.Lock();self.tasks={};self.events=[]

    def now(self):
        at=self.clock()
        if type(at) not in (int,float) or not math.isfinite(at) or at<0:raise ValueError('Measured nonnegative relative clock required')
        if self.events and at<self.events[-1]['at_s']:raise ValueError('Episode clock moved backwards')
        return at

    def event(self,kind,at,**fields):
        self.events.append(dict(type=kind,at_s=at,**copy.deepcopy(fields)))

    async def reserve_native_request(self,identity,arrived_user_transcript,*,minimum_delivery_delay_s=None):
        async with self.lock:
            at=self.now()
            if (not isinstance(identity,str) or not identity or identity in self.tasks
                    or len(self.tasks)>=self.maximum):raise ValueError('Distinct bounded actual native delegation identity required')
            context=transcript_snapshot(arrived_user_transcript,at)
            delay=self.delay if minimum_delivery_delay_s is None else minimum_delivery_delay_s
            if type(delay) not in (int,float) or not math.isfinite(delay) or not 0<=delay<=30:
                raise ValueError('Declare bounded transport delay before actual backend dispatch')
            self.tasks[identity]=dict(id=identity,revision=1,context=context,context_sha256=canonical_hash(context),
                native_request_observed_at_s=at,minimum_delivery_delay_s=delay,phase='reserved',cancelled=False,response=None)
            self.event('native_request_reserved',at,task_id=identity,context_sha256=canonical_hash(context),minimum_delivery_delay_s=delay)

    async def start_backend(self,identity,model):
        """Call immediately before actual transport; never while holding a model call."""
        async with self.lock:
            at=self.now();task=self.tasks[identity]
            if not isinstance(model,str) or not model.strip():raise ValueError('Explicit backend model required')
            if task['phase']!='reserved' or task['cancelled']:raise ValueError('Request already started or retired')
            active={k:dict(revision=v['revision'],phase=v['phase'],arrived_user_transcript=v['context'])
                    for k,v in self.tasks.items() if k!=identity and not v['cancelled']
                    and v['phase'] in ('executing','result_ready')}
            content=dict(type='string',pattern=r'\S')
            tools=[function('respond_to_user',dict(content=content),
                'Return the answer to the actual arrived user request. The voice model has already requested this reasoning work.')]
            if active:
                targets=dict(type='array',minItems=1,items=dict(type='string',enum=list(active)))
                tools.extend([function('cancel_reasoning',dict(task_ids=targets,content=content),
                    'Cancel delivery of identified pending requests, then return relevant requested speech. Remote computation is not aborted.'),
                    function('replace_reasoning',dict(task_ids=targets,content=content),
                    'Retire identified superseded requests and return the solved corrected request using the full arrived user history.')])
            payload=dict(model=model,max_output_tokens=4096,store=False,parallel_tool_calls=False,
                instructions='Help the native voice assistant with the latest actual user request. Solve it carefully using all relevant USER premises and corrections in the arrived transcript. Answer concisely in natural spoken English without Markdown, formulas or internal metadata. For a bundle optimization give the chosen item names, total cost and total points in at most twenty spoken words. Select one action. Cancel or replace only the identified relevant pending requests; preserve unrelated work. For cancellation return only the delivery-cancellation status; leave any accompanying basic arithmetic question for the voice assistant to answer itself. Tool status is committed by the application before content is delivered. Never claim remote execution was aborted. No scoring references or expected answers are available.',
                input=json.dumps(dict(arrived_user_transcript=task['context'],active_requests=active),sort_keys=True),
                tools=tools,tool_choice='required')
            task.update(phase='executing',backend_started_at_s=at,backend_model=model,
                        payload_sha256=canonical_hash(payload),allowed_target_ids=list(active))
            self.event('backend_transport_start',at,task_id=identity,payload_sha256=task['payload_sha256'])
            return copy.deepcopy(payload)

    async def fail_backend(self,identity,error_type):
        """Retire delivery after a failed/uncertain call; never retry or fake content."""
        async with self.lock:
            at=self.now();task=self.tasks[identity]
            if task['phase']!='executing' or task['response'] is not None:
                raise ValueError('Only the actual unresolved backend call can fail')
            if not isinstance(error_type,str) or not error_type:
                raise ValueError('Explicit nonsecret transport error type required')
            task.update(phase='transport_failed_preserved',error_type=error_type)
            self.event('backend_transport_failed',at,task_id=identity,error_type=error_type,
                       remote_execution_state_known=False,automatic_retry=False)

    async def complete_backend(self,identity,response,*,actual_transport):
        """Preserve a real completion even when a later cancellation hides it."""
        async with self.lock:
            at=self.now();task=self.tasks[identity]
            if type(actual_transport) is not bool:raise ValueError('Explicit actual/fixture transport classification required')
            if task['phase']!='executing' or task['response'] is not None:raise ValueError('No exact running backend completion to bind')
            task['response']=copy.deepcopy(response)
            task['provider_finished_at_s']=at;task['actual_transport']=actual_transport
            self.event('backend_transport_completed',at,task_id=identity,response_sha256=canonical_hash(response),
                       response_id=response.get('id'),actual_transport=actual_transport)
            calls=[v for v in response.get('output',[]) if v.get('type')=='function_call']
            if (response.get('status')!='completed' or response.get('model')!=task['backend_model']
                    or not isinstance(response.get('id'),str) or not response['id'] or len(calls)!=1):
                task['phase']='failed_preserved';raise ValueError('One actual completed provider action required')
            if any(t!=identity and v.get('response',{}).get('id')==response['id']
                   for t,v in self.tasks.items() if v.get('response') is not None):
                task['phase']='failed_preserved';raise ValueError('Provider response identity was reused across requests')
            call=calls[0]
            try:arguments=json.loads(call['arguments'])
            except (KeyError,ValueError,TypeError):
                task['phase']='failed_preserved';raise ValueError('Invalid provider action arguments') from None
            if not isinstance(arguments,dict):
                task['phase']='failed_preserved';raise ValueError('Provider arguments must be one semantic action object')
            operation=call.get('name');expected={'content'} if operation=='respond_to_user' else {'task_ids','content'}
            if (operation not in ('respond_to_user','cancel_reasoning','replace_reasoning')
                    or set(arguments)!=expected or not isinstance(arguments.get('content'),str)
                    or not arguments['content'].strip() or len(arguments['content'].split())>60):
                task['phase']='failed_preserved';raise ValueError('Invalid bounded semantic action; no targets changed')
            # A stale original completion remains in the raw ledger but cannot
            # mutate other tasks or become a returned native context span.
            if task['cancelled']:
                task['phase']='stale_completion_preserved'
                self.event('late_result_rejected',at,task_id=identity)
                return None
            targets=arguments.get('task_ids',[])
            if operation!='respond_to_user':
                if (not isinstance(targets,list) or not targets or any(not isinstance(t,str) for t in targets)
                        or len(set(targets))!=len(targets) or identity in targets
                        or not set(targets)<=set(task['allowed_target_ids'])
                        or any(t not in self.tasks or self.tasks[t]['cancelled']
                               or self.tasks[t]['phase'] not in ('executing','result_ready') for t in targets)):
                    task['phase']='failed_preserved';raise ValueError('Every selected target must still be pending; no partial retirement')
            retired=[dict(task_id=t,revision=self.tasks[t]['revision']) for t in targets]
            if operation=='replace_reasoning':task['revision']=1+max(self.tasks[t]['revision'] for t in targets)
            for t in targets:self.tasks[t]['cancelled']=True
            receipt=dict(task_id=identity,revision=task['revision'],operation=operation,
                actual_status_committed_at_s=at,retired_requests=retired,remote_execution_aborted=False,
                response_id=response['id'],model=response['model'],response_sha256=canonical_hash(response),
                request_payload_sha256=task['payload_sha256'],arrived_user_context_sha256=task['context_sha256'],
                backend_started_at_s=task['backend_started_at_s'],provider_finished_at_s=at,
                minimum_delivery_delay_s=task['minimum_delivery_delay_s'],
                delivery_eligible_at_s=max(at,task['backend_started_at_s']+task['minimum_delivery_delay_s']),
                actual_transport=actual_transport,content=arguments['content'])
            task.update(phase='result_ready',receipt=receipt)
            self.event('semantic_action_committed',at,task_id=identity,operation=operation,retired_requests=retired)
            return copy.deepcopy(receipt)

    async def publish(self,identity,send):
        """Serialize a bounded real wire write with cancellation commitment.

        No provider model call holds this lock. A write failure is uncertain and
        never retried: a duplicate commentary span could leak a stale result.
        Completion records local write time, not proof of server/model hearing.
        """
        async with self.lock:
            at=self.now();task=self.tasks[identity]
            if task['cancelled']:return False
            if task['phase']!='result_ready':raise ValueError('Only one current ready result may be published')
            receipt=task['receipt']
            if at<receipt['delivery_eligible_at_s']:return False
            payload=dict(type='session.commentary.append',event_id='reasoning_result_'+identity,
                         delegation_id=identity,content=receipt['content'])
            task['phase']='publishing'
            try:await asyncio.wait_for(send(copy.deepcopy(payload)),timeout=5)
            except BaseException:
                task['phase']='delivery_uncertain_no_retry';self.event('wire_write_failed_uncertain',self.now(),task_id=identity);raise
            done=self.now();task.update(phase='wire_sent',wire_write_completed_at_s=done)
            self.event('commentary_wire_write_completed',done,task_id=identity,server_or_model_arrival_confirmed=False)
            return True

    def evidence(self):
        return dict(schema='dl-native-reasoning-lifecycle-component-v1',tasks=copy.deepcopy(self.tasks),
            events=copy.deepcopy(self.events),API_calls_executed_by_this_module=0,
            native_request_selection_proven=False,teacher_or_training_approved=False,
            source_oracle_used=False,remote_execution_abort_supported=False,
            separate_raw_provider_voice_word_audio_semantic_conversion_and_native_audits_still_required=True)
