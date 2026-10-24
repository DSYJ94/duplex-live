"""Reject incomplete approvals and changed context-span acoustic ownership."""
from copy import deepcopy
import unittest

from evals.native_reasoning_gradient import validate_numerical


class ReasoningGradientGate(unittest.TestCase):
    def fixture(self):
        parent = {'checkpoint_sha256': 'a' * 64, 'optimizer_steps': 520}
        manifest = dict(complete32=True, passed=True, all_model_parameters_unchanged=True, cases=[])
        report = dict(schema='dl-native-reasoning32-numerical-v1', complete32=True, passed=True,
            all_model_parameters_unchanged=True, threshold=.001, causal_prefix_error_required=0.,
            optimizer_updates=0, new_checkpoint_written=False, training_approved=False,
            behavior_qualified=False, peak_allocated_bytes=40_000_000_000, parent=parent, cases=[])
        for i in range(32):
            delivered = i % 3
            manifest['cases'].append(dict(id=str(i), ownership=dict(text_targets=100, actual_delivered_contexts=delivered)))
            report['cases'].append(dict(id=str(i), passed=True, causal_prefix_maximum_error=0.,
                native_training_text_alignment_max_error=.0001, native_depth_parity={'passed': True},
                comparisons={k: {'maximum_absolute_error': .0001} for k in ('hidden', 'text_logits')},
                real_text_targets=100, real_acoustic_mask_frames=100-delivered,
                pre_span_acoustic_loss_exclusions=delivered,
                assistant_audio_target_counts=[100-delivered]+[99-delivered]*7, text_ce=1., acoustic_ce=2.))
        return report, manifest, parent

    def test_whole32_with_approved_distinct_text_audio_masks(self):
        report, manifest, parent = self.fixture()
        self.assertIs(validate_numerical(report, manifest, parent), report)

    def test_subset_reordering_duplicate_and_wrong_parent_rejected(self):
        for mode in ('subset', 'reorder', 'duplicate', 'parent', 'incomplete', 'memory'):
            report, manifest, parent = self.fixture()
            if mode == 'subset': report['cases'].pop()
            if mode == 'reorder': report['cases'].reverse()
            if mode == 'duplicate': report['cases'][-1] = deepcopy(report['cases'][0])
            if mode == 'parent': report['parent'] = {'checkpoint_sha256': 'b' * 64}
            if mode == 'incomplete': report['complete32'] = False
            if mode == 'memory': report['peak_allocated_bytes'] = 80_000_000_000
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_numerical(report, manifest, parent)

    def test_unmasked_or_extra_excluded_audio_frame_rejected(self):
        for count in (98, 100):
            report, manifest, parent = self.fixture()
            report['cases'][1]['real_acoustic_mask_frames'] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                validate_numerical(report, manifest, parent)

    def test_numerical_nan_causality_and_lost_head_rejected(self):
        for mode in ('nan', 'causality', 'alignment', 'depth', 'head', 'loss'):
            report, manifest, parent = self.fixture()
            row = report['cases'][19]
            if mode == 'nan': row['comparisons']['hidden']['maximum_absolute_error'] = float('nan')
            if mode == 'causality': row['causal_prefix_maximum_error'] = 1e-9
            if mode == 'alignment': row['native_training_text_alignment_max_error'] = .00101
            if mode == 'depth': row['native_depth_parity']['passed'] = False
            if mode == 'head': row['assistant_audio_target_counts'].pop()
            if mode == 'loss': row['acoustic_ce'] = float('inf')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_numerical(report, manifest, parent)


if __name__ == '__main__':
    unittest.main()
