"""Execute grounded backend plans asynchronously; the native model owns speech.

Planner and read-only lookup are explicit dependencies. This module contains no
provider credentials, speech policy, ASR router, evaluation answers or retries.
"""
import asyncio
import hashlib
import json
import time
import uuid
from .task_context import AudioEvidence
from .context_backend_plan import location_plan


class ContextPlanWorker:
    accepts_audio_evidence=True

    def __init__(self, planner, lookup, maximum_requests=6):
        if not callable(planner) or not callable(lookup) or type(maximum_requests) is not int or not 1<=maximum_requests<=6:
            raise ValueError('Explicit planner, read-only lookup and bounded requests required')
        self.planner,self.lookup,self.maximum=planner,lookup,maximum_requests
        self.tasks,self.states,self.events={}, {}, []
        self.closed=False

    def event(self,kind,**details):self.events.append(dict(type=kind,monotonic=time.monotonic(),**details))

    def submit(self,query,raw_query=None,audio_evidence=None):
        if self.closed:raise ValueError('Worker is closed')
        if not isinstance(query,str) or not query.strip() or len(query)>4000:
            self.event('native_delegate_rejected',reason='empty_or_unbounded_task')
            return None
        if type(audio_evidence) is not AudioEvidence:raise ValueError('Causal original audio required')
        if len(self.states)>=self.maximum:
            self.event('worker_request_limit',maximum_requests=self.maximum);return None
        ident=str(uuid.uuid4())
        self.states[ident]=dict(query=query,phase='reserved',cancelled=False,evidence=audio_evidence.receipt(),
                                plan_ready=asyncio.Event())
        self.event('native_delegate',work_id=ident,query=query,raw_query=query if raw_query is None else raw_query)
        self.tasks[ident]=asyncio.create_task(self.execute(ident,audio_evidence))
        return ident

    def finish(self,ident,**content):
        s=self.states[ident]
        if self.closed or s['cancelled']:
            s['phase']='stale_completion_preserved';return
        receipt=dict(work_id=ident,query=s['query'],audio_context_sha256=s['evidence']['pcm_sha256'],**content)
        receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
        s.update(phase='result_ready',receipt=receipt)
        self.event('worker_completed',**receipt)

    async def execute(self,ident,evidence):
        s=self.states[ident]
        if self.closed or s['cancelled']:return
        s['phase']='planning';self.event('worker_started',work_id=ident)
        try:
            response=await asyncio.wait_for(self.planner(s['query'],evidence),100)
            p=location_plan(response)
            if self.closed or s['cancelled']:return
            response_id=response.get('id')
            if (not isinstance(response_id,str) or not response_id or len(response_id)>256
                    or any(v.get('planner_response_id')==response_id for k,v in self.states.items() if k!=ident)):
                raise ValueError('Unique identified backend plan required')
            s['plan']=p;s['planner_response_id']=response.get('id')
            s['plan_ready'].set()
            intent,location=p['intent'],p['record_location']
            if intent=='lookup':
                s.update(phase='lookup_pending',record_location=location)
                value=await asyncio.wait_for(self.lookup(location),100)
                if not isinstance(value,str) or len(value)>2000:raise ValueError('Bounded actual record result required')
                self.finish(ident,result=value,operation='lookup',record_location=location,lookup_executed=True,planner_response_id=response_id)
            elif intent=='cancel':
                # A later cancellation plan may arrive before an earlier lookup
                # plan. Resolve older identities before claiming nothing is pending;
                # never target work submitted after this cancellation request.
                predecessors=list(self.states)[:list(self.states).index(ident)]
                if predecessors:
                    await asyncio.gather(*(self.states[k]['plan_ready'].wait() for k in predecessors))
                if self.closed or s['cancelled']:return
                pending=[k for k in predecessors if not (v:=self.states[k])['cancelled']
                         and v['phase'] in ('lookup_pending','result_ready') and 'record_location' in v]
                if location is not None:targets=[k for k in pending if self.states[k]['record_location']==location]
                elif len(pending)<=1:targets=pending
                else:
                    self.finish(ident,result='Which pending task should be cancelled?',operation='clarify',lookup_executed=False)
                    return
                # All identities are selected before mutation; unrelated work is retained.
                for k in targets:
                    self.states[k]['cancelled']=True
                    if self.tasks[k].done():
                        self.states[k]['phase']='delivery_retired'
                        self.event('result_delivery_retired',work_id=k)
                    else:self.tasks[k].cancel()
                if targets:await asyncio.gather(*(self.tasks[k] for k in targets),return_exceptions=True)
                self.event('cancellation_confirmed',work_id=ident,targets=targets,remote_execution_aborted=False)
                self.finish(ident,result='Cancelled pending lookup delivery.' if targets else 'No matching lookup was pending.',operation='cancel',cancelled_work_ids=targets,remote_execution_aborted=False)
            else:
                # Pause/resume are semantic feedback only. No host keyword mute,
                # forced speech or change to the native listen/speak policy.
                self.finish(ident,result=p['response'],operation=intent,lookup_executed=False)
        except asyncio.CancelledError:
            s['phase']='cancelled';self.event('worker_cancelled',work_id=ident,remote_execution_aborted=False)
            raise
        except Exception as error:
            self.finish(ident,status='request_failed',error_type=type(error).__name__,automatic_retry=False)
        finally:
            s['plan_ready'].set()

    def take_results(self):
        if self.closed:return None
        ready=[]
        for s in self.states.values():
            if s['phase']=='result_ready' and not s['cancelled']:
                ready.append(s['receipt']);s['phase']='context_queued'
        if not ready:return None
        return '<backend>'+json.dumps(ready,ensure_ascii=False).replace('<','\\u003c').replace('>','\\u003e')+'</backend>'

    async def close(self):
        self.closed=True
        for k,s in self.states.items():
            s['cancelled']=True
            if not self.tasks[k].done():self.tasks[k].cancel()
        if self.tasks:await asyncio.gather(*self.tasks.values(),return_exceptions=True)
