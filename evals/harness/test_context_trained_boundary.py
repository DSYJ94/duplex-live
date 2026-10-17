import unittest
from evals.context_trained_boundary import boundary


class TrainedBoundaryTest(unittest.TestCase):
    def scene(self,token=4):
        rows=[dict(text_token=3,input_start_s=i*.08) for i in range(30)]
        rows[7]['text_token']=0;rows[8]['text_token']=token
        text=[3,3,3,0,12,3,13,3,3]
        return rows,text

    def test_actual_first_word_and_ret_share_the_verified_prediction_delay(self):
        for token in [4,42]:
            rows,text=self.scene(token)
            self.assertEqual(boundary(rows,text,4,9,.2,.32),(7,12,3,token))

    def test_question_cannot_cross_the_next_user_project(self):
        rows,text=self.scene()
        self.assertEqual(boundary(rows,text,4,9,.2,.32,1.6),(7,12,3,4))
        with self.assertRaises(RuntimeError):boundary(rows,text,4,9,.2,.32,1.)

    def test_user_speech_and_actual_epad_cannot_be_overwritten(self):
        rows,text=self.scene()
        with self.assertRaises(RuntimeError):boundary(rows,text,4,9,.5,.32)
        rows[7]['text_token']=3
        with self.assertRaises(RuntimeError):boundary(rows,text,4,9,.2,.32)

    def test_past_words_invalid_teacher_and_clock_changes_rejected(self):
        rows,text=self.scene();rows[3]['text_token']=17
        with self.assertRaises(RuntimeError):boundary(rows,text,4,9,.2,.32)
        rows,text=self.scene();text[3]=3
        with self.assertRaises(RuntimeError):boundary(rows,text,4,9,.2,.32)
        rows,text=self.scene();rows[5]['input_start_s']+=.01
        with self.assertRaises(RuntimeError):boundary(rows,text,4,9,.2,.32)
        rows,text=self.scene()
        with self.assertRaises(RuntimeError):boundary(rows,text,4,9,.2,float('nan'))


if __name__=='__main__':unittest.main()
