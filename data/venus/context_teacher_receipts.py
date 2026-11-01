"""Execute local synthetic teacher work; no native dispatch/provider claims.

Controlled async gates define pending work for teacher construction. Cancel and
replace retire delivery, while the retired computation is drained and preserved.
These are backend receipts, not native histories or measured dialogue timings.
"""
import argparse,asyncio,hashlib,json,time,uuid
from pathlib import Path
from .context_targets import facts


def solve(scene,label,revision):
    f=facts(scene['worker_contract'])
    if label=='travel':return f['travel']
    if label=='meal':return dict(total_GBP=f['meal_total_GBP'],within_budget=f['meal_within_budget'])
    if label=='paint':return dict(litres=f['original_litres'] if revision==1 else f['revised_litres'],revision=revision)
    return f


async def collect(scene,draft):
    states={};current={};events=[];delivered=[];cancelled_labels=set()
    def event(kind,**details):events.append(dict(kind=kind,monotonic=time.monotonic(),**details))
    async def execute(ident):
        state=states[ident];event('worker_started',work_id=ident)
        await state['gate'].wait()
        result=solve(scene,state['label'],state['revision'])
        receipt=dict(work_id=ident,query=state['query'],revision=state['revision'],result=result,origin='executed_local_synthetic_contract_solver',remote_execution_aborted=False)
        receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest();state['receipt']=receipt
        event('worker_completed',work_id=ident,receipt=receipt)
        if state['retired']:event('stale_result_delivery_suppressed',work_id=ident)
        return receipt
    async def submit(action):
        label=action['task_label'];ident=str(uuid.uuid4());states[ident]=dict(label=label,query=action['text'],revision=action.get('revision',1),retired=False,gate=asyncio.Event())
        current[label]=ident;states[ident]['task']=asyncio.create_task(execute(ident));await asyncio.sleep(0)
        event('teacher_forced_dispatch',work_id=ident,task_label=label,revision=states[ident]['revision'])
    for action in draft['actions']:
        kind=action['kind'];label=action.get('task_label')
        if kind in ('cancel','replace'):
            ident=current[label];state=states[ident]
            if state['task'].done() or state['retired']:raise ValueError('Teacher cancellation requires genuine pending delivery')
            state['retired']=True;cancelled_labels.add(label);event('delivery_retired',work_id=ident,remote_execution_aborted=False)
            if kind=='replace':await submit(action)
        elif kind=='delegate':await submit(action)
        elif kind=='speak':
            if 'requires_cancel_receipt_for' in action and action['requires_cancel_receipt_for'] not in cancelled_labels:raise ValueError('Premature cancellation acknowledgement')
            if 'requires_pending' in action and states[current[action['requires_pending']]]['task'].done():raise ValueError('Unrelated work not pending')
            if 'requires_result_for' in action:
                ident=current[action['requires_result_for']];state=states[ident]
                if state['retired'] or state['revision']!=action.get('revision',1):raise ValueError('Wrong result revision')
                state['gate'].set();receipt=await state['task'];delivered.append(receipt)
                event('teacher_backend_result_available',work_id=ident)
            event('planned_spoken_target',text=action['text'],after_user_turn=action['after_user_turn'],audio_generated=False)
        elif kind=='pause':event('planned_silence',after_user_turn=action['after_user_turn'])
        else:raise ValueError('Unknown teacher action')
    for state in states.values():state['gate'].set()
    if states:await asyncio.gather(*(state['task'] for state in states.values()))
    if any(states[r['work_id']]['retired'] for r in delivered):raise ValueError('Retired result leaked into teacher feedback')
    return dict(scene_id=scene['id'],fold=scene['fold'],events=events,delivered_receipts=delivered,all_worker_receipts=[s['receipt'] for s in states.values()],teacher_forced_requests=True,native_student_dispatch_claimed=False,provider_calls=0,training_approved=False,native_timing_available=False)


async def main(a):
    if a.output.exists():raise ValueError('Preserve prior actual receipts')
    plan=json.loads(a.plan.read_text());draft=json.loads(a.draft.read_text());scenes={s['id']:s for s in plan['scenes']}
    rows=[await collect(scenes[d['id']],d) for d in draft['scenes']]
    assert len(rows)==36
    a.output.mkdir(parents=True,exist_ok=False)
    for r in rows:(a.output/(r['scene_id']+'.json')).write_text(json.dumps(r,indent=2)+'\n')
    report=dict(schema='dl-context-teacher-local-work-v1',complete36=True,scenes=rows,task_receipts=sum(len(r['all_worker_receipts']) for r in rows),delivered_receipts=sum(len(r['delivered_receipts']) for r in rows),native_teacher_histories_available=False,training_approved=False,provider_calls=0,optimizer_updates=0,plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),draft_sha256=hashlib.sha256(a.draft.read_bytes()).hexdigest())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='scenes'}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--draft',type=Path,required=True);p.add_argument('--output',type=Path,required=True);asyncio.run(main(p.parse_args()))
