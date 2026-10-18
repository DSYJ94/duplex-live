"""Real state mutation/receipts; model competence requires the GPU comparison."""
import copy
import unittest
from evals.idempotent_reasoning_worker import cancel_or_report, run_idempotent_reasoning_worker, require_idempotent_delivery
from evals.duplex_reasoning_worker import ReasoningWorld, run_reasoning_worker
from evals.recovery_worker import WorkerContractError
from evals.instruction_contract_probe import resume_fixture
from evals.harness.test_instruction_contract import Capture, cpu_executor


class IdempotentCancellationTest(unittest.TestCase):
    def world(self):
        world = ReasoningWorld({}); world.reserve('latest')
        for identity in ('active', 'finished', 'cancelled', 'unrelated'):
            world.reserve(identity); world.tasks[identity].update(operation='delegate_reasoning', phase='running')
        world.tasks['finished'].update(closed=True, phase='completed')
        world.tasks['cancelled']['cancelled'] = True
        return world

    def test_mixed_actual_states_report_truth_and_second_call_is_idempotent(self):
        world = self.world(); untouched = copy.deepcopy(world.tasks['unrelated'])
        receipt = cancel_or_report(world, ['active', 'finished', 'cancelled'], 'latest', 1, {t:1 for t in world.tasks})
        self.assertEqual(receipt['newly_cancelled_task_ids'], ['active'])
        self.assertEqual([r['status'] for r in receipt['receipts']], ['delivery_cancelled', 'already_finished', 'already_cancelled'])
        self.assertFalse(receipt['receipts'][1]['delivery_cancelled'])
        self.assertFalse(any(r['remote_execution_aborted'] for r in receipt['receipts']))
        again = cancel_or_report(world, ['active', 'finished', 'cancelled'], 'latest', 1, {t:1 for t in world.tasks})
        self.assertEqual(again['newly_cancelled_task_ids'], [])
        self.assertFalse(any(r['state_changed'] for r in again['receipts']))
        self.assertEqual(world.tasks['unrelated'], untouched)
        require_idempotent_delivery([receipt], world.tasks)
        require_idempotent_delivery([again], world.tasks)

    def test_unknown_reused_duplicate_or_current_ids_never_partially_mutate(self):
        for identities, revisions in [(['active','unknown'], {'active':1}), (['active','active'], {'active':1}),
                                      (['active','finished'], {'active':1,'finished':0}), (['latest'], {'latest':1})]:
            world = self.world(); before = copy.deepcopy(world.tasks)
            with self.assertRaises(ValueError):
                cancel_or_report(world, identities, 'latest', 1, revisions)
            self.assertEqual(world.tasks, before)

    def test_original_closed_state_rejects_operation_new_opt_in_returns_actual_noop(self):
        scene, unused = resume_fixture('unrelated-active', 0, 'fixture')
        def run(worker):
            world = ReasoningWorld({}); world.tasks = copy.deepcopy(scene['actual_world_tasks'])
            capture = Capture([dict(name='cancel_reasoning_and_resume_speech', arguments=dict(task_ids=['blue-count']))])
            result = worker(scene['context'], world, 'latest', 'fixture', 1, 'state-aware', scene['pending_speech'], cpu_executor(capture),
                resumable_speech=scene['resume_target'], bound_resume_target=True, allow_cancel_resume=True)
            return result, world
        with self.assertRaises(WorkerContractError):
            run(run_reasoning_worker)
        result, world = run(run_idempotent_reasoning_worker)
        self.assertEqual(result['route'], 'resume'); self.assertEqual(result['content'], '')
        self.assertEqual(result['resume_target'], scene['resume_target'])
        self.assertEqual(result['receipts'][0]['receipts'][0]['status'], 'already_finished')
        self.assertFalse(world.tasks['red-count']['cancelled'])
        self.assertFalse(world.tasks['blue-count']['cancelled'])
        self.assertTrue(result['requires_idempotent_media_receipt_guard'])
        require_idempotent_delivery(result['receipts'], world.tasks)

    def test_media_rejects_fabricated_status_revision_and_remote_abort_claim(self):
        world = self.world()
        good = cancel_or_report(world, ['active','finished'], 'latest', 1, {t:1 for t in world.tasks})
        for edit in ('status','revision','abort','changed','live-noop'):
            bad = copy.deepcopy(good)
            if edit=='status':bad['receipts'][1]['status']='delivery_cancelled'
            if edit=='revision':bad['receipts'][0]['revision']=8
            if edit=='abort':bad['remote_execution_aborted']=True
            if edit=='changed':bad['receipts'][1]['state_changed']=True
            if edit=='live-noop':
                bad['task_ids'][1]='unrelated';bad['receipts'][1].update(task_id='unrelated',closed=False)
            with self.assertRaises(WorkerContractError):require_idempotent_delivery([bad],world.tasks)

    def test_nonresume_native_request_has_exact_original_behavior(self):
        world = self.world(); capture = Capture([dict(name='answer', arguments=dict(content='Five.'))])
        base = cpu_executor(capture)
        result = run_idempotent_reasoning_worker([dict(role='user',text='What is two plus three?')], world, 'latest', 'fixture', 1, 'state-aware', [], base)
        self.assertEqual(result['content'], 'Five.'); self.assertEqual(result['receipts'], [])
        self.assertNotIn('reasoning_result_history', capture.inputs[0][0][0]['content'])


if __name__ == '__main__':
    unittest.main()
