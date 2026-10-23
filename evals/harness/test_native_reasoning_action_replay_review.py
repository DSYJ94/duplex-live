import copy
import unittest

from evals.harness import test_native_reasoning_action_replay as previous
from evals.native_reasoning_action_replay import restored_world
from evals.native_reasoning_action_replay_review import transition


class ComponentReadback(unittest.IsolatedAsyncioTestCase):
    async def fixture(self):
        parent=previous.ActualStateReplay().ledger();world,old,new,context=restored_world(parent)
        await world.reserve_native_request(new,context,minimum_delivery_delay_s=0.)
        payload=await world.start_backend(new,'gpt-6-luna')
        response=dict(id='fixture-response',model='gpt-6-luna',status='completed',output=[dict(
            type='function_call',name='replace_reasoning',arguments='{"task_ids":["original"],"content":"Corrected answer."}')])
        await world.complete_backend(new,response,actual_transport=True)
        return parent,world.evidence(),payload,response

    async def test_exact_frozen_state_receipt_is_recomputed_without_inventing_a_wire_write(self):
        args=await self.fixture();world=await transition(*args)
        self.assertTrue(world.tasks['original']['cancelled'])
        self.assertTrue(all('wire_write_completed_at_s'not in t for t in world.tasks.values()))

    async def test_payload_or_receipt_tampering_fails(self):
        for kind in ('payload','receipt'):
            args=list(await self.fixture())
            if kind=='payload':args[2]=dict(args[2],instructions='Unapproved action forcing.')
            else:
                args[1]=copy.deepcopy(args[1]);args[1]['tasks']['new']['receipt']['content']='Forged answer.'
            with self.subTest(kind=kind),self.assertRaises(ValueError):await transition(*args)


if __name__=='__main__':unittest.main()
