import unittest
from evals.native_old25_timing_repair import timing_only_words, REPAIRED


class TimingOnlyRepair(unittest.TestCase):
    def test_preserves_exact_original_labels_and_confidence(self):
        old = [dict(text='Okay,', start=.1, end=6.3, confidence=.8),
               dict(text="I'll", start=6.3, end=6.5, confidence=.9)]
        new = [dict(text='Okay.', start=6.2, end=6.4), dict(text="I'll", start=6.4, end=6.6)]
        got = timing_only_words(old, new)
        self.assertEqual([w['text'] for w in got], ['Okay,', "I'll"])
        self.assertEqual(got[0], dict(text='Okay,', start=6.2, end=6.4, confidence=.8))
        self.assertEqual(old[0]['start'], .1)

    def test_rejects_added_missing_reordered_or_substituted_words(self):
        old = [dict(text='orange', start=.1, end=.2), dict(text='square', start=.2, end=.4)]
        for texts in (['orange'], ['orange', 'square', 'square'], ['square', 'orange'], ['blue', 'square']):
            with self.subTest(texts=texts), self.assertRaises(RuntimeError):
                timing_only_words(old, [dict(text=t, start=6+i, end=6.5+i) for i,t in enumerate(texts)])

    def test_diagnosed_scope_is_fixed(self):
        self.assertEqual(len(REPAIRED), 4)
        self.assertIn('teacher-recovery-train-004', REPAIRED)


if __name__ == '__main__':
    unittest.main()
