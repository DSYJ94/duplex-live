"""Exercise mutation/guard/checkpoint ordering without Torch or a provider.

Synthetic protocol doubles below are NOT model runs or owner authorizations.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest

from evals import native_reasoning_continuation_loop as loop


class Operations:
    def __init__(self, checkpoint_fails=False):
        self.calls = []
        self.checkpoint_fails = checkpoint_fails

    def backward(self, index):
        self.calls.append(('backward', index))
        return dict(step=index + 1, global_adam_step=529 + index)

    def clip_and_check(self):
        self.calls.append(('clip',))
        return dict(pre_clip_gradient_norm=12.)

    def step(self):
        self.calls.append(('step',))

    def after_step(self, index):
        self.calls.append(('after', index))
        return dict(parameter_delta_l2=.1)

    def checkpoint(self, out, completed, row):
        self.calls.append(('checkpoint', completed))
        if self.checkpoint_fails:
            raise OSError('synthetic interrupted checkpoint')
        return dict(global_adam_step=row['global_adam_step'])

    def finish(self, rows):
        self.calls.append(('finish', len(rows)))
        return dict(synthetic_test_only=True)


class ContinuationLoop(unittest.TestCase):
    approved = dict(sha256='b' * 64)

    def guard(self):
        now = time.time()
        return dict(execution_authorized=True, live_provider_deadline_verified=True,
            live_external_cost_guard_verified=True, single_gpu_across_providers_verified=True,
            checked_unix=now, deadline_unix=now + 30000, provider='gcp',
            remaining_budget_usd=75, planned_maximum_cost_usd=60,
            resource_id=loop.RESOURCE_ID, authorization_sha256=self.approved['sha256'],
            optimizer_updates_authorized=8, parent_global_adam_step=528,
            final_global_adam_step_limit=536, native_start_unix=now - 60,
            hourly_price_usd=5.5,
            maximum_new_gpu_seconds=32400, prior_gpu_seconds_upper=40351,
            prior_scope_cost_upper_usd=91.98448055555555)

    def test_exactly_eight_mutations_and_checkpoints_never_claim_qualification(self):
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            ops = Operations()
            out = Path(name) / 'fit'
            result = loop._drive(ops, out, self.approved, self.guard)
            self.assertEqual(result['optimizer_step_calls_returned'], 8)
            self.assertEqual(result['closed_checkpoints'], 8)
            self.assertEqual([x['global_adam_step'] for x in result['rows']], list(range(529, 537)))
            self.assertFalse(result['native_restore_approved'])
            self.assertFalse(result['behavior_qualified'])
            self.assertFalse(result['model_adopted'])
            self.assertEqual(ops.calls[-1], ('finish', 8))
            self.assertEqual(sum(x == ('step',) for x in ops.calls), 8)
            self.assertTrue((out / 'training-record.json').exists())

    def test_guard_loss_after_backward_prevents_optimizer_mutation(self):
        count = 0
        def guard():
            nonlocal count
            count += 1
            value = self.guard()
            if count == 3:
                value['live_external_cost_guard_verified'] = False
            return value
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            ops = Operations()
            out = Path(name) / 'fit'
            with self.assertRaises(ValueError):
                loop._drive(ops, out, self.approved, guard)
            self.assertEqual(ops.calls, [('backward', 0)])
            events = [json.loads(x) for x in (out / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['optimizer_step_calls_returned'], 0)
            self.assertFalse((out / 'training-record.json').exists())

    def test_last_guard_after_clipping_can_still_prevent_mutation(self):
        count = 0
        def guard():
            nonlocal count
            count += 1
            value = self.guard()
            if count == 4:
                value['execution_authorized'] = False
            return value
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            ops = Operations()
            with self.assertRaises(ValueError):
                loop._drive(ops, Path(name) / 'fit', self.approved, guard)
            self.assertEqual(ops.calls, [('backward', 0), ('clip',)])

    def test_post_mutation_checkpoint_failure_preserves_actual_count_without_retry(self):
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            ops = Operations(checkpoint_fails=True)
            out = Path(name) / 'fit'
            with self.assertRaises(OSError):
                loop._drive(ops, out, self.approved, self.guard)
            events = [json.loads(x) for x in (out / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['optimizer_step_calls_returned'], 1)
            self.assertEqual(events[-1]['closed_checkpoints'], 0)
            self.assertTrue(events[-1]['optimizer_state_requires_audit'])
            self.assertEqual(sum(x == ('step',) for x in ops.calls), 1)
            self.assertNotIn(('backward', 1), ops.calls)
            with self.assertRaises(FileExistsError):
                loop._drive(Operations(), out, self.approved, self.guard)

    def test_optimizer_exception_never_implies_that_state_is_unchanged(self):
        class PartialMutation(Operations):
            def step(self):
                self.calls.append(('step',))
                raise RuntimeError('synthetic exception after partial mutation')
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            ops = PartialMutation()
            out = Path(name) / 'fit'
            with self.assertRaises(RuntimeError):
                loop._drive(ops, out, self.approved, self.guard)
            events = [json.loads(x) for x in (out / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['optimizer_step_calls_returned'], 0)
            self.assertTrue(events[-1]['optimizer_state_requires_audit'])
            self.assertEqual(sum(x == ('step',) for x in ops.calls), 1)

    def test_stale_reset_budget_old_scope_and_different_resource_fail(self):
        cases = [('checked_unix', time.time() - 31), ('prior_gpu_seconds_upper', 0),
                 ('prior_scope_cost_upper_usd', 0), ('optimizer_updates_authorized', 0),
                 ('parent_global_adam_step', 520), ('resource_id', 'different-vm'),
                 ('planned_maximum_cost_usd', 76), ('maximum_new_gpu_seconds', 300),
                 ('hourly_price_usd', 7.26), ('planned_maximum_cost_usd', 55),
                 ('authorization_sha256', 'c' * 64)]
        for key, value in cases:
            raw = self.guard()
            raw[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                loop.check_guard(lambda:raw, self.approved)

    def test_training_that_fits_alone_cannot_consume_validation_reserve(self):
        raw = self.guard()
        raw['deadline_unix'] = time.time() + 20000
        with tempfile.TemporaryDirectory() as name, redirect_stdout(io.StringIO()):
            ops = Operations()
            out = Path(name) / 'fit'
            with self.assertRaisesRegex(ValueError, 'Insufficient remaining time'):
                loop._drive(ops, out, self.approved, lambda:raw)
            self.assertEqual(ops.calls, [])
            self.assertFalse(out.exists())
        self.assertGreater(loop.required_completion_seconds(0), loop.required_completion_seconds(7))
        self.assertEqual(loop.required_completion_seconds(8), 13326)

    def test_actual_old_zero_update_grant_cannot_authorize_this_stage(self):
        root = Path(__file__).resolve().parents[2]
        with self.assertRaises(ValueError):
            loop.authorization(root / 'docs/GCP-NATIVE-CONTINUATION-AUTHORIZATION-2026-10-24.json')


if __name__ == '__main__':
    unittest.main()
