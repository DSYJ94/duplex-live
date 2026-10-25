import copy
import json
from pathlib import Path
import unittest

from portable.context.native_matched_reasoning_loss import summarize, diagnostic_deadline


class DiagnosticWindowTests(unittest.TestCase):
    def test_original_and_corrected_duration_retain_preservation_margin(self):
        self.assertEqual(diagnostic_deadline(100, 5500, 3600), 3700)
        self.assertEqual(diagnostic_deadline(100, 5500, 4800), 4900)
        self.assertEqual(diagnostic_deadline(100, 5020, 4800), 4900)

    def test_duration_cannot_shorten_or_exceed_explicit_bounds(self):
        for maximum in (True, 3600., 0, 3599, 3601, 4801, 5400):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                diagnostic_deadline(100, 10000, maximum)

    def test_short_or_nonfinite_provider_window_cannot_start(self):
        for end in (5019, float('nan'), float('inf')):
            with self.subTest(end=end), self.assertRaises(ValueError):
                diagnostic_deadline(100, end, 4800)


class MatchedReasoningLossTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        plan = json.loads((root/'evals/fixtures/native-reasoning-delegation-source-proposal-20261023.json').read_text())
        self.rows = [dict(id=r['id'], family=r['family'], text_ce=1., acoustic_ce=2.,
                          fingerprint=dict(native_text_calls_sha256='a'*64,
                                           assistant_audio_logits_sha256='b'*64,
                                           assistant_audio_targets=[3]*8, text_targets=4,
                                           audio_mask_frames=3)) for r in plan['cases']]
        self.expected = {r['id']: copy.deepcopy(r['fingerprint']) for r in self.rows}

    def test_complete_authored32_produces_all_four_family_means(self):
        result = summarize(self.rows, self.expected)
        self.assertEqual(result['overall'], dict(text_ce=1., acoustic_ce=2.))
        self.assertEqual(len(result['families']), 4)

    def test_missing_or_duplicated_scene_is_not_a_complete_endpoint(self):
        for rows in (self.rows[:-1], self.rows[:-1]+[self.rows[0]]):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                summarize(rows, self.expected)

    def test_unequal_family_coverage_is_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['family'] = rows[1]['family']
        with self.assertRaises(ValueError):
            summarize(rows, self.expected)

    def test_changed_saved_fingerprint_is_rejected(self):
        for key, value in [('audio_mask_frames', 2), ('native_text_calls_sha256', 'c'*64),
                           ('assistant_audio_targets', [2]*8)]:
            rows = copy.deepcopy(self.rows)
            rows[3]['fingerprint'][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                summarize(rows, self.expected)

    def test_nonfinite_or_invalid_scalar_is_rejected(self):
        for value in (float('nan'), float('inf'), -1., True):
            rows = copy.deepcopy(self.rows)
            rows[3]['text_ce'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(rows, self.expected)


if __name__ == '__main__':
    unittest.main()
