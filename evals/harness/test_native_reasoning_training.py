"""Whole32 exposure and context-span targets cannot silently become a subset."""
from copy import deepcopy
import unittest

from evals.native_reasoning_training import FAMILIES, schedule, exposure, validate_gradient


class ReasoningTraining(unittest.TestCase):
    def fixture(self):
        rows = [dict(id=f'{family}-{i}', family=family) for family in sorted(FAMILIES) for i in range(8)]
        teachers = schedule(rows)
        counts = {}
        for row in teachers:
            excluded = int(row['family'] != 'direct-basic')
            counts[row['id']] = dict(text=100, audio_mask_frames=100-excluded,
                pre_span_exclusions=excluded, audio=[100-excluded]+[99-excluded]*7)
        policy = dict(teachers=teachers, target_counts=counts,
            text_loss_targets_per8=sum(c['text'] for c in counts.values()),
            assistant_audio_loss_targets_per8=sum(sum(c['audio']) for c in counts.values()))
        steps = []
        for i in range(8):
            selected = []
            for row in teachers[i::8]:
                c = counts[row['id']]
                selected.append(dict(id=row['id'], family=row['family'], weight=row['weight_when_selected'],
                    text_targets=c['text'], audio_mask_frames=c['audio_mask_frames'], text_ce=1., acoustic_ce=2.,
                    acoustic_objective=dict(assistant_audio_targets=c['audio'], user_audio_loss_targets=0),
                    prefix_span_user_audio_loss_targets=0))
            steps.append(dict(step=i+1, reasoning32_microbatch=selected))
        return policy, steps

    def test_all_families_every_update_and_all32_exactly_once(self):
        policy, steps = self.fixture()
        self.assertTrue(exposure(steps, policy))
        self.assertEqual(sum(t['weight_per_epoch'] for t in policy['teachers']), 1.)
        for step in steps:
            self.assertEqual(sum(r['weight'] for r in step['reasoning32_microbatch']), 1.)

    def test_subset_duplicate_changed_family_or_epoch_weight_rejected(self):
        for mode in ('subset', 'duplicate', 'family', 'weight', 'epoch_weight'):
            policy, steps = self.fixture()
            row = steps[0]['reasoning32_microbatch'][0]
            if mode == 'subset': steps[-1]['reasoning32_microbatch'].pop()
            if mode == 'duplicate': steps[-1]['reasoning32_microbatch'][0] = deepcopy(row)
            if mode == 'family': row['family'] = 'direct-basic'
            if mode == 'weight': row['weight'] /= 2
            if mode == 'epoch_weight': policy['teachers'][0]['weight_per_epoch'] /= 2
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                exposure(steps, policy)

    def test_extra_masking_unmasked_span_user_loss_and_bad_totals_rejected(self):
        for mode in ('mask_more', 'mask_less', 'head', 'user', 'prefix', 'total', 'nan'):
            policy, steps = self.fixture()
            row = steps[0]['reasoning32_microbatch'][0]
            if mode == 'mask_more': row['audio_mask_frames'] -= 1
            if mode == 'mask_less': row['audio_mask_frames'] += 1
            if mode == 'head': row['acoustic_objective']['assistant_audio_targets'] = [99]*8
            if mode == 'user': row['acoustic_objective']['user_audio_loss_targets'] = 1
            if mode == 'prefix': row['prefix_span_user_audio_loss_targets'] = 1
            if mode == 'total': policy['assistant_audio_loss_targets_per8'] -= 1
            if mode == 'nan': row['text_ce'] = float('nan')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                exposure(steps, policy)

    def test_gradient_approval_requires_complete_finite_and_matching_ownership(self):
        from evals.harness.test_native_reasoning_gradient import ReasoningGradientGate
        numeric, manifest, parent = ReasoningGradientGate().fixture()
        ready = dict(schema='dl-native-reasoning32-gradient-readiness-v1', passed=True, complete32=True,
            all_parameters_and_execution_unchanged=True, trainable_parameters=1894400,
            optimizer_updates=0, checkpoint_written=False, training_approved=False, behavior_qualified=False,
            peak_allocated_bytes=40_000_000_000, parent=parent,
            native_execution_profile='BF16 native recorded-call autograd; compile and CUDA graphs disabled as in retained trainer', cases=[])
        for n in numeric['cases']:
            ready['cases'].append(dict(**{k:n[k] for k in ('id','real_text_targets','real_acoustic_mask_frames','pre_span_acoustic_loss_exclusions')},
                passed=True, exact_native_streamed_text_outputs=True, exact_native_acoustic_outputs=True,
                combined_gradient=dict(acoustic_text_head_gradient_delta_exactly_zero=True,
                    all26_gradients_present_and_finite=True, frozen_parameter_gradients=0,
                    combined_gradient_norm=1., acoustic_temporal_gradient_delta_norm=.2),
                acoustic_objective=dict(assistant_audio_targets=n['assistant_audio_target_counts'], user_audio_loss_targets=0)))
        self.assertIs(validate_gradient(ready,manifest,numeric,parent), ready)
        for mode in ('subset','changed_mask','frozen','dead_acoustic'):
            changed = deepcopy(ready)
            if mode == 'subset': changed['cases'].pop()
            if mode == 'changed_mask': changed['cases'][0]['real_acoustic_mask_frames'] -= 1
            if mode == 'frozen': changed['cases'][0]['combined_gradient']['frozen_parameter_gradients'] = 1
            if mode == 'dead_acoustic': changed['cases'][0]['combined_gradient']['acoustic_temporal_gradient_delta_norm'] = 0.
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_gradient(changed,manifest,numeric,parent)


if __name__ == '__main__':
    unittest.main()
