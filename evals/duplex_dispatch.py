"""Separate unknown semantic decisions from actually dispatched background work."""


def dispatch_status(world,metadata):
    unresolved=[];background=[];retired=[]
    with world.lock:
        for entry in metadata:
            identity=entry['task_id'];state=world.tasks.get(identity)
            if state is None:raise RuntimeError('Pending dispatch has no actual world reservation')
            if state['revision']!=entry['revision'] or state['cancelled'] or state['closed']:
                retired.append(identity)
            elif state.get('operation') in ('lookup_record','delegate_reasoning'):
                # Only validated application execution sets this field. A model
                # plan, transcript or speculative tool name is not sufficient.
                background.append(identity)
            else:unresolved.append(identity)
    return dict(unresolved_decisions=unresolved,actual_background_tasks=background,retired_tasks=retired)
