import copy
import unittest
from evals.context_response_boundary import boundary


class ResponseBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.rows=[{'text_token':3,'input_start_s':i*.08} for i in range(100)]
        self.rows[49]['text_token']=0
        self.rows[50]['text_token']=4
        self.text=[3]*60+[0,12,3,13,3,3]+[3]*34

    def check(self,**kwargs):
        return boundary(self.rows,self.text,61,66,kwargs.get('last_user',3.6),
                        kwargs.get('first_onset',4.88))

    def test_complete_reply_moves_to_observed_cue_with_spacing_retained(self):
        self.assertEqual(self.check(),(49,54,-12))

    def test_first_word_replaces_ret_prediction_without_repeating_cached_epad(self):
        # Engine returns text one frame after predicting it. Returned RET50
        # was predicted at49; EPAD was already predicted at48 and is in cache.
        self.assertEqual(boundary(self.rows,self.text,61,66,3.6,4.88),(49,54,-12))

    def test_no_interruption_of_unfinished_user(self):
        with self.assertRaises(RuntimeError):self.check(last_user=3.9)

    def test_existing_sampled_reply_cannot_be_erased(self):
        self.rows[30]['text_token']=17
        with self.assertRaises(RuntimeError):self.check()

    def test_missing_native_speaking_cue_is_rejected(self):
        self.rows[49]['text_token']=3
        with self.assertRaises(RuntimeError):self.check()

    def test_repeating_teacher_epad_instead_of_first_word_is_rejected(self):
        with self.assertRaises(RuntimeError):boundary(self.rows,self.text,60,66,3.6,4.88)

    def test_no_teacher_content_may_be_omitted(self):
        for frame in (30,80):
            with self.subTest(frame=frame):
                original=copy.deepcopy(self.text);self.text[frame]=14
                with self.assertRaises(RuntimeError):self.check()
                self.text=original

    def test_invalid_clocks_or_exhausted_recording_are_rejected(self):
        for changes in ({'first_onset':float('nan')},{'last_user':float('inf')}):
            with self.assertRaises(RuntimeError):self.check(**changes)
        self.rows[9]['input_start_s']+=.01
        with self.assertRaises(RuntimeError):self.check()
        self.setUp();self.rows=self.rows[:52]
        with self.assertRaises(RuntimeError):self.check()


if __name__=='__main__':unittest.main()
