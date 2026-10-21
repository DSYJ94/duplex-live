"""Finite TRAIN spelling policy must never equate different spoken words."""
import unittest
from evals.native_control_words import agreement, spoken_words


class SpokenOrthography(unittest.TestCase):
    def test_same_word_spelling_retains_raw_error(self):
        result = agreement('Water vapour cools.', 'Water vapor cools.')
        self.assertEqual(result['raw_wer'], 1 / 3)
        self.assertEqual(result['canonical_word_wer'], 0.)
        self.assertEqual(result['variants'], {'vapor': 'vapour'})

    def test_names_cues_omissions_and_repetitions_still_fail(self):
        for a, b in [('Mia', 'Nia'), ('Beacon', 'Deacon'), ('room sign', 'rune sign'),
                     ('Beacon. Every basket holds apples.', 'Every basket holds apples.'),
                     ('Mia came first.', 'Mia came first. Mia came first.')]:
            with self.subTest(reference=a, actual=b):
                self.assertNotEqual(spoken_words(a), spoken_words(b))
                self.assertGreater(agreement(a, b)['canonical_word_wer'], 0.)


if __name__ == '__main__':
    unittest.main()
