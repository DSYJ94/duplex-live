"""Boundary ordering for the expanded native runner; caller owns collectives."""
from pathlib import Path
from .adaptive_policy import Policy
from .adaptive_evidence import build_request,backup,read


def context(checkpoint,receipt,data):
    step,checkpoint_hash,backup_hash=backup(receipt)
    metadata,_=read(str(checkpoint)+'.sha256.json')
    if metadata['sha256']!=checkpoint_hash or metadata['step']!=step:
        raise ValueError('Decision must bind the just-published native checkpoint')
    _,data_hash=read(data)
    return dict(resume_step=step,checkpoint_sha256=checkpoint_hash,backup_sha256=backup_hash,data_sha256=data_hash)


def request_for(plan,root,expanded,retained,receipt,step,multiplier):
    parent=Path(plan['parent_root']);root=Path(root);boundary=step//49*49
    def dev(n):
        name='dev-step-%04d.json'%n
        return root/name if (root/name).exists()else parent/name
    # Complete historical log prefixes are read; saved updates are never run.
    logs=sorted(set(parent.glob('job-*/train/rank-*/training.jsonl'))|
                set(root.glob('job-*/train/rank-*/training.jsonl')))
    return build_request(native=Path(expanded)/'inputs.json',retained=Path(retained)/'inputs.json',
        dev=dev(boundary),previous_dev=dev(boundary-49),backup_receipt=receipt,logs=logs,
        multiplier=multiplier,policy=Policy())


def publish_decision(controller,*,checkpoint,backup_checkpoint,context_value,request,step,write_receipt):
    """Callbacks run on all ranks; rank zero alone authors the request.

    This is also exercised without a voice model to enforce crash-safe order.
    Pre-decision context already binds a durable independently backed state.
    """
    if not controller.state['enabled']:controller.enable(context_value)
    record=controller.evaluate(request,context_value)
    if record.get('status')=='rejected':raise ValueError('Native decision rejected; preserve predecessor and stop')
    path=checkpoint(step,'adaptive-step-%04d.pt'%step)
    receipt=backup_checkpoint(path)
    write_receipt(dict(step=step,decision=record,checkpoint=str(path),backup_receipt=str(receipt),new_optimizer_updates=0))
    return path
