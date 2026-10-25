"""The proposed continuation must retain all losses and use NEW Adam clocks."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evals import native_reasoning_continuation_objective as objective
from evals.harness import test_context_native_complete_batch as complete
from evals.harness import test_context_native_stream_training as streaming
from evals.harness import test_native_control_training as control
from evals.harness import test_native_reasoning_training as reasoning
from evals.native_control_training import schedule as control_schedule


class ContinuationObjective(unittest.TestCase):
    def fixture(self):
        groups = complete.CompleteBalancedBatch().groups()
        for group in groups.values():
            for row in group:
                row['loss'] = 1.
        native = streaming.NativeStreamTraining()
        native_policy = native.declared()
        native_rows = native.steps(native_policy)
        for step in native_rows:
            for row in step['native_stream_teacher_microbatch']:
                row['balanced_spoken_clock_ce'] = .5
        ctrl = control.ControlTraining()
        control_policy = dict(teachers=control_schedule(ctrl.records()))
        control_rows = ctrl.steps(control_policy)
        reasoning_policy, rows = reasoning.ReasoningTraining().fixture()
        for i, row in enumerate(rows):
            row.update(native_rows[i])
            row.update(control_rows[i])
            row.update(global_adam_step=529 + i,
                       complete_balanced_microbatches=deepcopy(groups),
                       assistant_audio49_microbatch=deepcopy(groups['audio49']),
                       additional_native_v2_microbatch=deepcopy(groups['new24']),
                       learning_rate=1e-6, all26_gradients_finite=True,
                       frozen_parameter_gradients=0, pre_clip_gradient_norm=12.,
                       gradient_clip_norm=1.)
            row['loss_record'] = objective.loss_record(row)
        audio_policy = dict(teacher_ids=[r['id'] for r in groups['audio49']],
                            source_unique49_acoustic_targets=49 * 16,
                            complete_batch_membership=groups,
                            native_stream_auxiliary=native_policy)
        return rows, audio_policy, control_policy, reasoning_policy

    def test_complete_new_stage_does_not_approve_restore_or_behavior(self):
        result = objective.validate_stage(*self.fixture())
        self.assertEqual(result['final_global_adam_step'], 536)
        self.assertFalse(result['numerical_restore_approved'])
        self.assertFalse(result['behavior_qualified'])
        self.assertFalse(result['model_adopted'])

    def test_sum_matches_backward_coefficients_including_anchors_and_audio(self):
        rows, _, _, _ = self.fixture()
        report = rows[0]['loss_record']
        self.assertEqual(report['retained154']['word'], 64.)
        self.assertEqual(report['retained154']['clock'], 64.)
        self.assertAlmostEqual(sum(report['retained154'].values()), 133.)
        self.assertAlmostEqual(report['supplemental']['control40_text'], 1.)
        self.assertAlmostEqual(report['supplemental']['control40_acoustic'], 2.)
        self.assertAlmostEqual(report['supplemental']['reasoning32_text'], 1.)
        self.assertAlmostEqual(report['supplemental']['reasoning32_acoustic'], 2.)
        expected_native = sum(r['weight'] * .5 for r in rows[0]['native_stream_teacher_microbatch'])
        self.assertAlmostEqual(report['sampled_pre_update_objective'], 139. + expected_native)
        self.assertFalse(report['fixed_complete_endpoint'])

    def test_subset_replay_clock_lr_and_frozen_gradient_failures(self):
        for mode in ('old25', 'native49', 'control40', 'reasoning32', 'old_clock',
                     'duplicate_clock', 'wrong_rate', 'frozen', 'nan', 'forged_loss'):
            rows, *policies = self.fixture()
            row = rows[-1]
            if mode == 'old25': row['complete_balanced_microbatches']['old25'].pop()
            if mode == 'native49': row['native_stream_teacher_microbatch'].pop()
            if mode == 'control40': row['control40_microbatch'].pop()
            if mode == 'reasoning32': row['reasoning32_microbatch'].pop()
            if mode == 'old_clock': row['global_adam_step'] = 528
            if mode == 'duplicate_clock': row['global_adam_step'] = 535
            if mode == 'wrong_rate': row['learning_rate'] = 2e-7
            if mode == 'frozen': row['frozen_parameter_gradients'] = 1
            if mode == 'nan': row['pre_clip_gradient_norm'] = float('nan')
            if mode == 'forged_loss': row['loss_record']['retained154']['actual'] = 0.
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                objective.validate_stage(rows, *policies)

    def test_composition_reuses_original_backward_functions_in_order(self):
        rows, audio, _, _ = self.fixture()
        sizes = dict(batches=28, retention=4, references=4, native_teacher_batches=25,
                     additional_batches=24, warm_references=10, clock_references=14,
                     audio49_batches=49, stream_bound=49, control_bound=40, reasoning_bound=32)
        context = {name: [object()] * count for name, count in sizes.items()}
        context.update(lm=object(), component_kinds=[], response_boundaries=[],
                       audits=[], forward=object(), audio49_policy=audio,
                       args=SimpleNamespace(actual_response_first_word_weight=16))
        called = []
        def result(name, value):
            def run(*args):
                called.append((name, args))
                return deepcopy(value)
            return run
        with patch('evals.context_native_complete_batch.backward',
                   side_effect=result('complete', rows[0]['complete_balanced_microbatches'])), \
             patch('evals.context_native_stream_training.backward',
                   side_effect=result('native', rows[0]['native_stream_teacher_microbatch'])), \
             patch('evals.native_control_training.backward',
                   side_effect=result('control', rows[0]['control40_microbatch'])), \
             patch('evals.native_reasoning_training.backward',
                   side_effect=result('reasoning', rows[0]['reasoning32_microbatch'])):
            row = objective.backward(context, 0)
        self.assertEqual([name for name, _ in called], ['complete', 'native', 'control', 'reasoning'])
        self.assertEqual(called[0][1][-1], 16)
        self.assertEqual([args[-1] for _, args in called[1:]], [0, 0, 0])
        self.assertEqual(row['global_adam_step'], 529)
        self.assertEqual(row['loss_record'], rows[0]['loss_record'])

    def test_bad_step_or_incomplete_context_rejects_before_backward(self):
        for step in (True, -1, 8, 528):
            with self.subTest(step=step), self.assertRaises(ValueError):
                objective.backward({}, step)
        sizes = dict(batches=28, retention=4, references=4, native_teacher_batches=24,
                     additional_batches=24, warm_references=10, clock_references=14,
                     audio49_batches=49, stream_bound=49, control_bound=40, reasoning_bound=32)
        context = {name: [None] * count for name, count in sizes.items()}
        with patch('evals.context_native_complete_batch.backward') as backward:
            with self.assertRaises(ValueError):
                objective.backward(context, 0)
            backward.assert_not_called()


if __name__ == '__main__':
    unittest.main()
