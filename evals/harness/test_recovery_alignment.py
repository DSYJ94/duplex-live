import unittest
from evals.recovery_prepare_data import spoken_words


class SpokenSpelling(unittest.TestCase):
    def test_same_spoken_words_accept_both_spellings(self):
        self.assertEqual(spoken_words("The lookup is cancelled."),spoken_words("The lookup is canceled."))
        self.assertEqual(spoken_words("Sound is travelling."),spoken_words("Sound is traveling."))

    def test_different_spoken_words_still_fail(self):
        self.assertNotEqual(spoken_words("The lookup is cancelled."),spoken_words("The lookout is cancelled."))
        self.assertNotEqual(spoken_words("The lookup is cancelled."),spoken_words("The lookup is complete."))


if __name__ == "__main__":
    unittest.main()
