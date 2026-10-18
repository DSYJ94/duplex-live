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


def require_cancelled_delivery(receipts,tasks):
    if not isinstance(receipts,list) or len(receipts)!=1:raise WorkerContractError('Resume cancellation needs its actual delivery receipt',{})
    receipt=receipts[0];ids=receipt.get('task_ids')
    if (receipt.get('tool')!='cancel_reasoning' or receipt.get('delivery_cancelled') is not True or receipt.get('remote_execution_aborted') is not False
            or not isinstance(ids,list) or not ids or any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids)
            or any(i not in tasks or not tasks[i]['cancelled'] or tasks[i].get('operation')!='delegate_reasoning' for i in ids)
            or len(receipt.get('receipts',[]))!=len(ids)
            or any(r.get('task_id')!=i or r.get('tool')!='cancel_reasoning' or r.get('delivery_cancelled') is not True or r.get('remote_execution_aborted') is not False for i,r in zip(ids,receipt['receipts']))):
        raise WorkerContractError('Resume cancellation is not backed by actual task state',dict(receipts=receipts))
