import unittest
from evals.duplex_pause_confirmation import PauseConfirmation


class ConfirmationTest(unittest.TestCase):
    def test_noise_and_punctuation_cannot_pause_existing_speech(self):
        p=PauseConfirmation()
        for _ in range(100):
            self.assertFalse(p.update('',True));self.assertFalse(p.update('?',True))
        self.assertTrue(p.update('please',True))

    def test_one_pause_for_real_words_then_rearm_only_at_completed_turn(self):
        p=PauseConfirmation();self.assertTrue(p.update('could',True))
        self.assertFalse(p.update('you',True));self.assertFalse(p.update('',False))
        self.assertFalse(p.update('continue',True));self.assertFalse(p.update('',False,True))
        self.assertTrue(p.update('stop',True))

    def test_late_hallucinated_words_without_recent_sound_do_not_pause(self):
        p=PauseConfirmation();self.assertFalse(p.update('hello',False));self.assertTrue(p.update('hello',True))

    def test_boundaries_do_not_infer_intent_or_audibility(self):
        for word in ('no','yes','thanks','stop','continue','banana'):
            self.assertTrue(PauseConfirmation().update(word,True))

if __name__=='__main__':unittest.main()
