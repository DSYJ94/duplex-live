import unittest
from evals.recovery_prepare_data import spoken_words
from evals.recovery_encode import native_word_timing


class SpokenSpelling(unittest.TestCase):
    def test_same_spoken_words_accept_both_spellings(self):
        self.assertEqual(spoken_words("The lookup is cancelled."),spoken_words("The lookup is canceled."))
        self.assertEqual(spoken_words("Sound is travelling."),spoken_words("Sound is traveling."))

    def test_different_spoken_words_still_fail(self):
        self.assertNotEqual(spoken_words("The lookup is cancelled."),spoken_words("The lookout is cancelled."))
        self.assertNotEqual(spoken_words("The lookup is cancelled."),spoken_words("The lookup is complete."))


class NativeTiming(unittest.TestCase):
    def test_actual_first_subword_frames(self):
        words = [["alpha",[.08,.3],"SPEAKER_MAIN"],["beta",[.32,.5],"SPEAKER_MAIN"]]
        vocab = {"alpha":[5,6],"beta":[7]}
        report = native_word_timing([4,5,6,3,7],words,vocab.__getitem__,12.5)
        self.assertEqual(report["max_first_word_offset_ms"],0.)
        self.assertEqual(report["word_count"],2)

    def test_late_native_words_cannot_pass_word_grid_alone(self):
        words = [["alpha",[.08,.3],"SPEAKER_MAIN"]]
        with self.assertRaisesRegex(RuntimeError,"200ms"):
            native_word_timing([3]*10+[5,6],words,lambda _: [5,6],12.5)

    def test_native_words_cannot_disappear_or_reorder(self):
        words = [["alpha",[.08,.3],"SPEAKER_MAIN"]]
        for codes in ([3,6,5],[3,5]):
            with self.assertRaisesRegex(RuntimeError,"dropped or reordered"):
                native_word_timing(codes,words,lambda _: [5,6],12.5)


if __name__ == "__main__":
    unittest.main()
