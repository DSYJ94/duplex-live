"""Identity boundary for resuming a retained speech generator.

The controller chooses whether to resume. These checks establish only that the
named generator exists; they never infer intent or a physically heard cursor.
"""
from evals.recovery_worker import WorkerContractError


def validate_target(target,pending):
    if (not isinstance(target,dict) or set(target)!={'speech_id','event_id'}
            or any(not isinstance(v,str) or not v for v in target.values())
            or target['speech_id'] not in {p['id'] for p in pending}):
        raise WorkerContractError('Resume target lacks an actual owed retained generator',dict(resume_target=target))
    return dict(target)


def require_retained(target,active,paused):
    if (paused is None or active is None or active['result']['route']!='answer'
            or target!={'speech_id':active['task_id'],'event_id':active['event_id']}):
        raise WorkerContractError('Retained generator changed before semantic pause decision',dict(resume_target=target))
