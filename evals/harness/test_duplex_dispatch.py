from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from evals.recovery_worker import RecordWorld
from evals.duplex_reasoning_worker import ReasoningWorld
from evals.duplex_dispatch import dispatch_status


class SemanticDecisionBarrier(unittest.TestCase):
    def test_unknown_decision_blocks_but_actual_read_does_not_hold_foreground_speech(self):
        world=RecordWorld({'Cedar':{'label':'violet'}});revision=world.reserve('read')
        pending=[dict(task_id='read',revision=revision)]
        self.assertEqual(dispatch_status(world,pending)['unresolved_decisions'],['read'])
        world.read('read','Cedar','label',revision)
        self.assertEqual(dispatch_status(world,pending)['unresolved_decisions'],[])
        self.assertEqual(dispatch_status(world,pending)['actual_background_tasks'],['read'])
        second=world.reserve('new-user');pending.append(dict(task_id='new-user',revision=second))
        self.assertEqual(dispatch_status(world,pending)['unresolved_decisions'],['new-user'])
    def test_actual_running_reasoner_does_not_block_but_unresolved_foreground_still_does(self):
        began=threading.Event();release=threading.Event()
        def backend(question):
            began.set();self.assertTrue(release.wait(2));return dict(content='Actual fixture answer',remote_api_used=True)
        world=ReasoningWorld({},backend);revision=world.reserve('background');pending=[dict(task_id='background',revision=revision)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(world.reason,'background',revision,'Actual user question')
            try:
                self.assertTrue(began.wait(1));self.assertFalse(future.done())
                self.assertEqual(dispatch_status(world,pending)['unresolved_decisions'],[])
                second=world.reserve('new-user');pending.append(dict(task_id='new-user',revision=second))
                self.assertEqual(dispatch_status(world,pending)['unresolved_decisions'],['new-user'])
                world.finish('new-user',second);self.assertEqual(dispatch_status(world,pending)['unresolved_decisions'],[])
            finally:release.set()
            future.result()
    def test_missing_reservation_rejects_and_retired_work_does_not_hold_speech(self):
        world=RecordWorld({})
        with self.assertRaises(RuntimeError):dispatch_status(world,[dict(task_id='absent',revision=1)])
        revision=world.reserve('old');world.cancel('old')
        status=dispatch_status(world,[dict(task_id='old',revision=revision)])
        self.assertEqual(status,dict(unresolved_decisions=[],actual_background_tasks=[],retired_tasks=['old']))
