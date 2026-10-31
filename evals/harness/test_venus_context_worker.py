import asyncio
import json
import itertools
import pytest
from portable.venus.context_worker import ContextPlanWorker
from portable.venus.task_context import AudioEvidence


_ids=itertools.count()
def response(intent,location=None):
    return dict(id='fixture-plan-'+str(next(_ids)),status='completed',output=[dict(type='function_call',name='plan_context_task_v2',arguments=json.dumps(dict(intent=intent,record_location=location,grounding='audio',response='Fixture feedback.')))])


def submit(worker,query):return worker.submit(query,audio_evidence=AudioEvidence(0,0,bytes(32000)))


@pytest.mark.anyio
async def test_real_pending_lookup_selective_cancellation_and_unrelated_delivery():
    started={x:asyncio.Event() for x in ('alpha','beta')};gates={x:asyncio.Event() for x in started}
    async def planner(q,e):return response('cancel' if q=='cancelalpha' else 'lookup','alpha' if q in ('alpha','cancelalpha') else 'beta')
    async def lookup(location):
        started[location].set();await gates[location].wait();return 'Actual entry '+location
    w=ContextPlanWorker(planner,lookup)
    a=submit(w,'alpha');b=submit(w,'beta')
    await started['alpha'].wait();await started['beta'].wait()
    cancel=submit(w,'cancelalpha');await w.tasks[cancel]
    assert w.states[a]['phase']=='cancelled' and not w.states[b]['cancelled']
    assert any(e['type']=='worker_cancelled' and e['work_id']==a for e in w.events)
    receipt=w.take_results();assert 'Cancelled pending' in receipt and 'Actual entry' not in receipt
    gates['beta'].set();await w.tasks[b]
    assert 'Actual entry beta' in w.take_results() and w.take_results() is None
    await w.close()


@pytest.mark.anyio
async def test_missing_cancel_target_does_not_cancel_multiple_unrelated_tasks():
    gate=asyncio.Event();started=asyncio.Event();n=0
    async def planner(q,e):return response('cancel',None) if q=='cancel' else response('lookup',q)
    async def lookup(location):
        nonlocal n
        n+=1
        if n==2:started.set()
        await gate.wait();return location
    w=ContextPlanWorker(planner,lookup);a=submit(w,'alpha');b=submit(w,'beta');await started.wait()
    c=submit(w,'cancel');await w.tasks[c]
    assert not w.states[a]['cancelled'] and not w.states[b]['cancelled']
    assert 'Which pending task' in w.take_results()
    await w.close()


@pytest.mark.anyio
async def test_bad_plan_and_empty_request_cannot_execute_lookup():
    called=[]
    async def planner(*args):return dict(status='incomplete',output=[])
    async def lookup(location):called.append(location);return 'Never'
    w=ContextPlanWorker(planner,lookup)
    assert submit(w,'') is None
    ident=submit(w,'task');await w.tasks[ident]
    result=w.take_results();assert 'request_failed' in result and 'result' not in json.loads(result[9:-10])[0]
    assert not called
    await w.close()


@pytest.mark.anyio
async def test_cancel_retires_ready_but_uninjected_result_without_claiming_compute_abort():
    async def planner(q,e):return response('cancel' if q=='cancel' else 'lookup','alpha')
    async def lookup(location):return 'Already computed entry'
    w=ContextPlanWorker(planner,lookup);a=submit(w,'alpha');await w.tasks[a]
    b=submit(w,'cancel');await w.tasks[b]
    receipt=w.take_results()
    assert 'Already computed entry' not in receipt and 'Cancelled pending' in receipt
    assert w.states[a]['phase']=='delivery_retired'
    assert not any(x['type']=='worker_cancelled' for x in w.events)
    assert any(x['type']=='result_delivery_retired' for x in w.events)
    await w.close()


@pytest.mark.anyio
async def test_cancellation_plan_can_arrive_before_older_lookup_plan():
    release_plan=asyncio.Event();cancel_planned=asyncio.Event();hold_lookup=asyncio.Event()
    async def planner(q,e):
        if q=='lookup':await release_plan.wait()
        else:cancel_planned.set()
        return response('cancel' if q=='cancel' else 'lookup','alpha')
    async def lookup(location):await hold_lookup.wait();return 'Must not be delivered'
    w=ContextPlanWorker(planner,lookup)
    a=submit(w,'lookup');b=submit(w,'cancel');await cancel_planned.wait()
    assert not w.tasks[b].done()
    release_plan.set();await asyncio.wait_for(w.tasks[b],1)
    assert w.states[a]['cancelled'] and w.states[a]['phase']=='cancelled'
    receipt=w.take_results()
    assert 'Cancelled pending' in receipt and 'Must not be delivered' not in receipt
    await w.close()
