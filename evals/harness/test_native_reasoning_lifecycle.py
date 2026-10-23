import asyncio
import copy
import json
import itertools
import re
import unittest
from evals.native_reasoning_lifecycle import NativeReasoningLifecycle


IDS=itertools.count()


def response(name='respond_to_user',**arguments):
    return dict(status='completed',id='fixture-response-'+str(next(IDS)),model='fixture-only',
                output=[dict(type='function_call',name=name,arguments=json.dumps(arguments))])


class NativeReasoningLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.at=1.;self.world=NativeReasoningLifecycle(lambda:self.at,16.)
        self.context=[dict(speaker='user',text='Actual original problem.',received_at_s=.5)]

    async def reserve(self,identity,context=None):
        await self.world.reserve_native_request(identity,self.context if context is None else context)
        return await self.world.start_backend(identity,'fixture-only')

    async def test_request_is_immutable_arrived_prefix_and_contains_no_operator_annotations(self):
        payload=await self.reserve('original');self.context.append(dict(speaker='user',text='Future correction.',received_at_s=2.))
        self.assertNotIn('Future correction',payload['input'])
        self.assertEqual(self.world.tasks['original']['context'][0]['text'],'Actual original problem.')
        with self.assertRaises(ValueError):await self.world.reserve_native_request('bad',self.context)
        with self.assertRaises(ValueError):await self.world.reserve_native_request('bad',[dict(self.context[0],oracle='answer')])
        self.assertNotIn('expected',payload['input']);self.assertNotIn('oracle',payload['input'])
        pattern=payload['tools'][0]['parameters']['properties']['content']['pattern']
        self.assertIsNotNone(re.search(pattern,'Actual spoken answer.'));self.assertIsNone(re.search(pattern,'   '))

    async def test_cancel_runs_while_old_backend_is_pending_and_late_actual_return_is_preserved(self):
        await self.reserve('original');self.at=2.;await self.reserve('cancel')
        self.at=3.;r=await self.world.complete_backend('cancel',response('cancel_reasoning',task_ids=['original'],content='The new basic answer.'),actual_transport=False)
        self.assertEqual(r['retired_requests'],[dict(task_id='original',revision=1)])
        self.assertFalse(r['remote_execution_aborted']);self.at=4.
        self.assertIsNone(await self.world.complete_backend('original',response(content='Stale result.'),actual_transport=False))
        self.assertEqual(self.world.tasks['original']['phase'],'stale_completion_preserved')
        async def forbidden(_):raise AssertionError('Stale result was sent')
        self.assertFalse(await self.world.publish('original',forbidden))

    async def test_replacement_retains_full_actual_user_history_and_revises_only_selected_work(self):
        await self.reserve('original');await self.reserve('unrelated');self.at=2.
        history=self.context+[dict(speaker='user',text='Change only the budget.',received_at_s=1.5)]
        payload=await self.reserve('replacement',history)
        self.assertIn('Actual original problem.',payload['input']);self.assertIn('Change only the budget.',payload['input'])
        r=await self.world.complete_backend('replacement',response('replace_reasoning',task_ids=['original'],content='Corrected solution.'),actual_transport=False)
        self.assertEqual(r['revision'],2);self.assertTrue(self.world.tasks['original']['cancelled'])
        self.assertFalse(self.world.tasks['unrelated']['cancelled'])

    async def test_invalid_multi_target_action_cannot_partially_cancel_real_pending_work(self):
        await self.reserve('original');self.at=2.;await self.reserve('cancel')
        with self.assertRaises(ValueError):
            await self.world.complete_backend('cancel',response('cancel_reasoning',task_ids=['original','unknown'],content='Cancelled.'),actual_transport=False)
        self.assertFalse(self.world.tasks['original']['cancelled'])

    async def test_provider_completion_and_transport_delay_and_wire_time_are_separate(self):
        await self.reserve('original');self.at=3.
        r=await self.world.complete_backend('original',response(content='Actual answer.'),actual_transport=False)
        self.assertEqual(r['provider_finished_at_s'],3.);self.assertEqual(r['delivery_eligible_at_s'],17.)
        sent=[]
        async def send(payload):sent.append(payload);self.at+=.1
        self.assertFalse(await self.world.publish('original',send));self.assertEqual(sent,[])
        self.at=17.;self.assertTrue(await self.world.publish('original',send));self.assertEqual(len(sent),1)
        with self.assertRaises(ValueError):await self.world.publish('original',send)
        self.assertFalse(self.world.events[-1]['server_or_model_arrival_confirmed'])

    async def test_uncertain_wire_failure_never_retries_or_claims_delivery(self):
        await self.reserve('original');self.at=20.
        await self.world.complete_backend('original',response(content='Answer.'),actual_transport=False)
        async def broken(_):raise ConnectionError('fixture transport uncertainty')
        with self.assertRaises(ConnectionError):await self.world.publish('original',broken)
        self.assertEqual(self.world.tasks['original']['phase'],'delivery_uncertain_no_retry')
        with self.assertRaises(ValueError):await self.world.publish('original',broken)

    async def test_replayed_provider_identity_cannot_be_rebound_to_another_request(self):
        await self.reserve('original');saved=response(content='Original answer.')
        await self.world.complete_backend('original',saved,actual_transport=False)
        self.at=2.;await self.reserve('other')
        with self.assertRaises(ValueError):await self.world.complete_backend('other',saved,actual_transport=False)
        self.assertEqual(self.world.tasks['other']['phase'],'failed_preserved')
        self.assertFalse(self.world.tasks['original']['cancelled'])


if __name__=='__main__':unittest.main()
