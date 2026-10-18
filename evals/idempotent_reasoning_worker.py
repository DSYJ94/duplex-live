"""Opt-in fixture experiment: truthful idempotent cancel plus retained resume.

The model selects actual task IDs. This never infers intent from user words,
repairs generated output, or reports finished work as newly cancelled. The
production media receipt guard does not yet support this protocol.
"""
import copy
import json
import time
from evals.duplex_reasoning_worker import run_reasoning_worker
from evals.recovery_worker import WorkerContractError, WorkerSuperseded, tool


def cancel_or_report(world, task_ids, current_task, current_revision, target_revisions):
    """Validate the entire target set before any mutation, under the world lock."""
    with world.lock:
        current = world.tasks.get(current_task)
        if not current or current['revision'] != current_revision or current['closed'] or current['cancelled']:
            raise WorkerSuperseded('Current cancellation request is no longer active')
        if (not isinstance(task_ids, list) or not task_ids or any(not isinstance(t, str) for t in task_ids)
                or len(set(task_ids)) != len(task_ids) or current_task in task_ids):
            raise ValueError('Expected distinct actual reasoning result identities')
        for identity in task_ids:
            state = world.tasks.get(identity)
            if (identity not in target_revisions or not state or state.get('operation') != 'delegate_reasoning'
                    or state['revision'] != target_revisions[identity]):
                raise ValueError('Unknown or revised reasoning result; nothing changed')
        active = [t for t in task_ids if not world.tasks[t]['closed'] and not world.tasks[t]['cancelled']]
        status = {t: 'already_cancelled' if world.tasks[t]['cancelled'] else 'already_finished'
                  for t in task_ids if t not in active}
        changed = world.cancel_delivery(active) if active else None
        for identity in active:
            status[identity] = 'delivery_cancelled'
        receipts = [dict(task_id=t, revision=world.tasks[t]['revision'], status=status[t],
            state_changed=t in active, delivery_cancelled=world.tasks[t]['cancelled'],
            closed=world.tasks[t]['closed'], remote_execution_aborted=False) for t in task_ids]
        return dict(tool='idempotent_cancel_reasoning_and_resume_speech', task_ids=task_ids,
            newly_cancelled_task_ids=active, receipts=receipts, actual_cancellation_receipt=changed,
            remote_execution_aborted=False, new_speech_generated=False)


def run_idempotent_reasoning_worker(context, world, task_id, model, expected_revision, dispatch_protocol,
                                    pending_speech, response_executor, **options):
    if options.get('resumable_speech') is None or not options.get('allow_cancel_resume'):
        return run_reasoning_worker(context, world, task_id, model, expected_revision, dispatch_protocol,
            pending_speech, response_executor, **options)
    class Capture:
        provider_name = 'payload-capture-only-no-model-or-operation'
        def __call__(self, payload):
            self.payload = copy.deepcopy(payload)
            return dict(id='capture-only', model=model, status='completed', output=[dict(type='function_call',
                name='respond_to_user', arguments=json.dumps(dict(route='answer', content='Internal payload capture only.', consume_pending_speech_ids=[])))])
    recorder = Capture()
    with world.lock:
        run_reasoning_worker(context, world, task_id, model, expected_revision, dispatch_protocol,
            pending_speech, recorder, **options)
        history = {k: {name: copy.deepcopy(v[name]) for name in ('revision', 'operation', 'phase', 'question', 'closed', 'cancelled') if name in v}
            for k, v in world.tasks.items() if k != task_id and v.get('operation') == 'delegate_reasoning'}
    if not history:
        return run_reasoning_worker(context, world, task_id, model, expected_revision, dispatch_protocol,
            pending_speech, response_executor, **options)
    payload = copy.deepcopy(recorder.payload)
    target = options['resumable_speech']; bound = options.get('bound_resume_target', False)
    parameters = {} if bound else dict(speech_id=dict(type='string', enum=[target['speech_id']]), event_id=dict(type='string', enum=[target['event_id']]))
    parameters['task_ids'] = dict(type='array', minItems=1, items=dict(type='string', enum=sorted(history)))
    definition = tool('cancel_reasoning_and_resume_speech',
        'Request delivery cancellation for the identified reasoning results AND continue the sole retained paused speech generator, without any new words. '
        'This operation is idempotent: an active target is actually cancelled; a finished or already-cancelled target returns its actual unchanged status. '
        'Use the actual reasoning_result_history to identify the user-requested targets, never unrelated work. '
        'It never claims finished work was newly cancelled or remote computation was aborted. Retained audio resumes after the actual status/cancellation receipt.', parameters)
    payload['tools'] = [t for t in payload['tools'] if t['name'] != definition['name']] + [definition]
    payload['local_worker_state']['reasoning_result_history'] = history
    before = time.monotonic(); response = response_executor(payload)
    calls = [c for c in response.get('output', []) if c.get('type') == 'function_call']
    if response.get('status') != 'completed' or response.get('model') != model or len(calls) != 1:
        raise WorkerContractError('Idempotent trial requires one actual native action', dict(response=response))
    if calls[0]['name'] != definition['name']:
        class Replay:
            provider_name = response_executor.provider_name
            def __call__(self, new):
                if new != recorder.payload:
                    raise WorkerContractError('World changed before original action dispatch', dict(response=response))
                return response
        return run_reasoning_worker(context, world, task_id, model, expected_revision, dispatch_protocol,
            pending_speech, Replay(), **options)
    arguments = json.loads(calls[0]['arguments']); expected = {} if bound else target
    if set(arguments) != set(expected) | {'task_ids'} or {k: v for k, v in arguments.items() if k != 'task_ids'} != expected:
        raise WorkerContractError('Idempotent action lacks the exact retained generator', dict(response=response))
    try:
        receipt = cancel_or_report(world, arguments['task_ids'], task_id, expected_revision,
            {k: v['revision'] for k, v in history.items()})
    except ValueError as error:
        raise WorkerContractError('Idempotent targets rejected before mutation: '+str(error), dict(response=response)) from error
    event = dict(response_id=response['id'], model=response['model'], provider=response_executor.provider_name,
        usage=response.get('usage'), seconds=time.monotonic()-before,
        local_inference=response.get('local_inference'), output=response['output'])
    return dict(route='resume', content='', consume_pending_speech_ids=[], receipts=[receipt], events=[event],
        task_id=task_id, resume_target=copy.deepcopy(target), resume_after_idempotent_cancellation=True,
        speech_model_routing=False, final_response_route='resume', dispatch_protocol=dispatch_protocol,
        response_provider=response_executor.provider_name, actual_reasoning_backend=None,
        agent_actions_executed=False, cancellation_supported='idempotent-delivery-only',
        remote_execution_abort_supported=False, production_media_receipt_guard_supported=False)
