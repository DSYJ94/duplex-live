"""Hold older delivery while a newer native-selected state decision is unresolved.

No intent/word classifier: only actual request order and the explicit older task
IDs offered to a backend decision. Provider execution, deliberate transport hold
and this causal delivery barrier remain distinct. Failed decisions fail closed.
"""
import asyncio
import copy

from evals.native_reasoning_lifecycle import NativeReasoningLifecycle as OriginalLifecycle


class NativeReasoningLifecycle(OriginalLifecycle):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._last_blockers={}

    def _blockers(self,identity):
        later=False;blocked=[]
        for key,task in self.tasks.items():
            if key==identity:later=True;continue
            if not later or task['cancelled']:continue
            if task['phase']=='reserved':blocked.append(key)
            elif (task['phase'] in ('executing','failed_preserved','transport_failed_preserved')
                  and identity in task.get('allowed_target_ids',[])):blocked.append(key)
        return blocked

    def _delivery_state(self,identity):
        at=self.now();task=self.tasks[identity]
        if task['cancelled']:return 'retired'
        if task['phase']!='result_ready':raise ValueError('Only a current ready result may be published')
        if at<task['receipt']['delivery_eligible_at_s']:return 'not_due'
        blockers=self._blockers(identity);previous=self._last_blockers.get(identity,[])
        if blockers!=previous:
            self._last_blockers[identity]=list(blockers)
            self.event('delivery_barrier_wait' if blockers else 'delivery_barrier_released',at,
                       task_id=identity,pending_newer_decision_ids=blockers,
                       provider_computation_relabelled=False)
        return 'pending_decision' if blockers else 'ready'

    async def delivery_state(self,identity):
        async with self.lock:return self._delivery_state(identity)

    def evidence(self):
        value=super().evidence()
        value.update(schema='dl-native-reasoning-lifecycle-component-v2',
                     delivery_barrier='Unresolved newer native-selected state decisions hold older delivery; no intent classifier',
                     failed_uncertain_decisions_fail_closed=True)
        return value

    async def publish(self,identity,send):
        """Recheck state atomically with the same bounded one-time wire write."""
        async with self.lock:
            if self._delivery_state(identity)!='ready':return False
            at=self.now();task=self.tasks[identity];receipt=task['receipt']
            payload=dict(type='session.commentary.append',event_id='reasoning_result_'+identity,
                         delegation_id=identity,content=receipt['content'])
            task['phase']='publishing'
            try:await asyncio.wait_for(send(copy.deepcopy(payload)),timeout=5)
            except BaseException:
                task['phase']='delivery_uncertain_no_retry';self.event('wire_write_failed_uncertain',self.now(),task_id=identity);raise
            done=self.now();task.update(phase='wire_sent',wire_write_completed_at_s=done)
            self.event('commentary_wire_write_completed',done,task_id=identity,server_or_model_arrival_confirmed=False)
            return True
