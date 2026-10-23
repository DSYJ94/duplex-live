import copy
import unittest

from evals.native_reasoning_action_replay import restored_world


class ActualStateReplay(unittest.IsolatedAsyncioTestCase):
    def ledger(self):
        context=[dict(speaker='user',text='Original premise.',received_at_s=1.)]
        return dict(tasks={'original':dict(id='original',revision=1,context=context,phase='wire_sent',
            cancelled=False,response={'id':'old-response'},receipt={'content':'Old actual content.'},
            wire_write_completed_at_s=20.),'new':dict(native_request_observed_at_s=10.,
            context=context+[dict(speaker='user',text='Changed constraint.',received_at_s=9.)])})

    async def test_reconstruction_is_explicit_and_never_modifies_the_saved_parent(self):
        ledger=self.ledger();before=copy.deepcopy(ledger);world,old,new,context=restored_world(ledger)
        self.assertEqual(ledger,before);self.assertEqual(world.tasks[old]['phase'],'result_ready')
        self.assertNotIn('wire_write_completed_at_s',world.tasks[old])
        self.assertEqual(world.tasks[old]['response'],ledger['tasks'][old]['response'])
        await world.reserve_native_request(new,context)
        payload=await world.start_backend(new,'fixture-model')
        self.assertEqual(world.tasks[new]['allowed_target_ids'],[old])
        self.assertNotIn('oracle',payload['input']);self.assertNotIn('expected',payload['input'])
        self.assertFalse(world.evidence()['native_request_selection_proven'])

    async def test_only_one_new_request_fits_the_explicit_replay_budget(self):
        world,_,new,context=restored_world(self.ledger())
        await world.reserve_native_request(new,context)
        with self.assertRaises(ValueError):await world.reserve_native_request('extra',context)


if __name__=='__main__':unittest.main()
