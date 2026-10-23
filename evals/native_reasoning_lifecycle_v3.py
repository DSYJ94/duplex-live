"""Explicit backend action contract, preserving V2 causal delivery barriers.

The model still chooses independent response, cancellation or replacement from
actual arrived USER history and pending identities. No intent classifier,
case-specific routing, oracle, response rewriting or automatic retry.
"""
import copy
import json

from evals.native_reasoning_lifecycle import canonical_hash, function
from evals.native_reasoning_lifecycle_v2 import NativeReasoningLifecycle as PreviousLifecycle

ACTION_CONTRACT = (
    'Choose the application action before composing the spoken answer. '
    'respond_to_user preserves every existing active request: use it only for independent new work '
    'that neither changes nor withdraws an active request. '
    'A correction, refinement or changed constraint on an active request MUST use replace_reasoning, '
    'with the relevant active task_ids, even when you can already compute the corrected answer. '
    'Returning a corrected answer through respond_to_user is incorrect because the obsolete result '
    'would remain deliverable. A user withdrawal MUST use cancel_reasoning for the relevant task_ids. '
    'An active result_ready request is still pending delivery and can be replaced or cancelled; '
    'its computation being finished does not mean the user has heard it. '
    'Preserve unrelated active work. Use only the actual arrived USER premises and corrections. '
    'Solve carefully. Give concise natural spoken English, without Markdown or metadata. '
    'For a bundle answer give item names, total cost and total points in at most twenty words. '
    'A cancellation returns only the cancellation status; the voice assistant handles accompanying '
    'basic arithmetic. Never claim remote computation stopped. No expected answers are supplied.'
)


class NativeReasoningLifecycle(PreviousLifecycle):
    async def start_backend(self,identity,model):
        async with self.lock:
            at=self.now();task=self.tasks[identity]
            if not isinstance(model,str) or not model.strip():raise ValueError('Explicit backend model required')
            if task['phase'] != 'reserved' or task['cancelled']:raise ValueError('Request already started or retired')
            active={k:dict(revision=v['revision'],phase=v['phase'],arrived_user_transcript=v['context'])
                    for k,v in self.tasks.items()if k != identity and not v['cancelled']
                    and v['phase']in ('executing','result_ready')}
            content=dict(type='string',pattern=r'\S')
            tools=[function('respond_to_user',dict(content=content),
                'Answer independent new work only. Preserves ALL active requests. A correction of an active task must use replace_reasoning instead.')]
            if active:
                targets=dict(type='array',minItems=1,items=dict(type='string',enum=list(active)))
                tools.extend([function('cancel_reasoning',dict(task_ids=targets,content=content),
                    'Commit withdrawal of relevant active work before confirming. Retires delivery only; remote computation is not aborted.'),
                    function('replace_reasoning',dict(task_ids=targets,content=content),
                    'Required for any correction, refinement or constraint change to relevant active work. Retire its old result and return the solved corrected task.')])
            payload=dict(model=model,max_output_tokens=4096,store=False,parallel_tool_calls=False,
                         instructions=ACTION_CONTRACT,
                         input=json.dumps(dict(arrived_user_transcript=task['context'],active_requests=active),sort_keys=True),
                         tools=tools,tool_choice='required')
            task.update(phase='executing',backend_started_at_s=at,backend_model=model,
                        payload_sha256=canonical_hash(payload),allowed_target_ids=list(active))
            self.event('backend_transport_start',at,task_id=identity,payload_sha256=task['payload_sha256'],
                       action_contract='explicit-independent-vs-pending-correction-v1')
            return copy.deepcopy(payload)

    def evidence(self):
        value=super().evidence()
        value.update(schema='dl-native-reasoning-lifecycle-component-v3',
                     backend_action_contract='explicit-independent-vs-pending-correction-v1',
                     semantic_actions_model_selected=True,intent_or_case_classifier=False)
        return value
