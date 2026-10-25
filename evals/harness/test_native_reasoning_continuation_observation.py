"""Observer binding/comparison is separate from native GPU parity acceptance."""
from copy import deepcopy
import inspect
from pathlib import Path
import sys
import unittest

from evals.native_reasoning_continuation_observation import compare, load_observer


class ContinuationObservation(unittest.TestCase):
    def test_new_observer_does_not_replace_original_training_module(self):
        from evals import native_reasoning_training as original
        root = Path(__file__).resolve().parents[2]
        observed = load_observer(root)
        self.assertIs(sys.modules['evals.native_reasoning_training'], original)
        self.assertIsNot(observed, original.fingerprints)
        self.assertEqual(inspect.signature(observed).parameters['metrics'].kind,
                         inspect.Parameter.KEYWORD_ONLY)
        self.assertIn('metrics', inspect.getsource(observed))

    def test_measured_comparison_keeps_process_identity_separate(self):
        before = dict(all_fingerprints=dict(original28={'case':'a' * 64}),
                      original28_metrics=[dict(nll=.5)],
                      reasoning32_metrics=[dict(text_ce=1., acoustic_ce=.4)],
                      reasoning32_summary=dict(overall=dict(text_ce=1.)),
                      process_identity=dict(pid=1))
        restored = deepcopy(before)
        restored['process_identity'] = dict(pid=2)
        self.assertTrue(compare(restored, before))
        for key in ('all_fingerprints', 'original28_metrics', 'reasoning32_metrics', 'reasoning32_summary'):
            bad = deepcopy(restored)
            bad[key] = {}
            with self.subTest(key=key), self.assertRaises(ValueError):
                compare(bad, before)


if __name__ == '__main__':
    unittest.main()
