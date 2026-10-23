"""Counterexample from actual cancellation: the state-decision call can be slow."""
import unittest
from evals.native_reasoning_lifecycle import NativeReasoningLifecycle as Original
from evals.native_reasoning_lifecycle_v2 import NativeReasoningLifecycle as Revised
from evals.harness.test_native_reasoning_lifecycle import response


class DecisionBarrier(unittest.IsolatedAsyncioTestCase):
    async def setup_world(self,kind=Revised):
        self.at=1.;world=kind(lambda:self.at,0.);context=[dict(speaker='user',text='Actual original task.',received_at_s=.5)]
        await world.reserve_native_request('old',context);await world.start_backend('old','fixture-only')
        await world.complete_backend('old',response(content='Original computed answer.'),actual_transport=False)
        self.at=2.;await world.reserve_native_request('new',context+[dict(speaker='user',text='Actual update.',received_at_s=1.5)])
        return world

    async def test_reserved_and_executing_new_request_hold_old_without_word_routing(self):
        world=await self.setup_world();sent=[]
        async def send(p):sent.append(p)
        self.assertEqual(await world.delivery_state('old'),'pending_decision')
        self.assertFalse(await world.publish('old',send));await world.start_backend('new','fixture-only')
        self.at=20.;self.assertFalse(await world.publish('old',send));self.assertEqual(sent,[])
        await world.complete_backend('new',response('cancel_reasoning',task_ids=['old'],content='Delivery cancelled.'),actual_transport=False)
        self.assertEqual(await world.delivery_state('old'),'retired');self.assertFalse(await world.publish('old',send))
        self.assertTrue(await world.publish('new',send));self.assertEqual([p['delegation_id'] for p in sent],['new'])

    async def test_unrelated_answer_releases_original_after_decision_without_cancelling_it(self):
        world=await self.setup_world();await world.start_backend('new','fixture-only');self.at=3.
        await world.complete_backend('new',response(content='Separate computed answer.'),actual_transport=False)
        self.assertEqual(await world.delivery_state('old'),'ready');self.assertFalse(world.tasks['old']['cancelled'])
        sent=[]
        async def send(p):sent.append(p)
        self.assertTrue(await world.publish('old',send));self.assertEqual(sent[0]['content'],'Original computed answer.')

    async def test_failed_uncertain_decision_never_releases_old_result(self):
        world=await self.setup_world();await world.start_backend('new','fixture-only');self.at=3.
        await world.fail_backend('new','ReadTimeout')
        self.assertEqual(await world.delivery_state('old'),'pending_decision')
        async def forbidden(_):raise AssertionError('Uncertain cancellation released old result')
        self.assertFalse(await world.publish('old',forbidden))

    async def test_old_ledger_reproduces_actual_race_and_rejects_late_cancel(self):
        world=await self.setup_world(Original);await world.start_backend('new','fixture-only');self.at=20.
        async def send(_):pass
        self.assertTrue(await world.publish('old',send))
        with self.assertRaisesRegex(ValueError,'pending'):
            await world.complete_backend('new',response('cancel_reasoning',task_ids=['old'],content='Delivery cancelled.'),actual_transport=False)


if __name__=='__main__':unittest.main()
