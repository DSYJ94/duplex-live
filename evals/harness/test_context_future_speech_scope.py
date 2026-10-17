import unittest
from evals.context_future_speech_scope import approved_words
from evals.harness.test_context_clarification_future import QuestionScopeTests


class FutureSpeechScopeTest(unittest.TestCase):
    def test_partial_question_does_not_claim_the_later_fact(self):
        actual,teacher,side=QuestionScopeTests().scene()
        m={'target_kind':'clarification_question','teacher':teacher,
            'all_teacher_reply_and_request_tokens_retained':False,
            'later_teacher_reply_and_request_tokens_deliberately_excluded':True,
            'clarification_question_without_backend_context':True,
            'actual_private_receipt':None,'future_recorded_receipts':[],
            'question_only_selection':{'source_first_word_index':0,'source_end_word_index':1,'source_end_frame':78},
            'future_end_real_frame':78}
        words,scope=approved_words(m,side,actual)
        self.assertEqual(words,[side['alignments'][0][0]])
        self.assertEqual(scope,'first_clarification_question_only')
        m['question_only_selection']['source_end_word_index']=2
        with self.assertRaises(RuntimeError):approved_words(m,side,actual)

    def test_original_full_reply_guard_still_rejects_an_implicit_crop(self):
        side={'alignments':[['The answer.',[1.,2.],'SPEAKER_MAIN']]}
        with self.assertRaises(RuntimeError):approved_words({'all_teacher_reply_and_request_tokens_retained':False},side,{})
        self.assertEqual(approved_words({'all_teacher_reply_and_request_tokens_retained':True},side,{}),
            (['The answer.'],'complete_teacher_reply'))

    def test_trained_question_retiming_keeps_the_same_complete_question_before_project(self):
        actual,teacher,side=QuestionScopeTests().scene()
        m={'target_kind':'clarification_question','teacher':teacher,
            'all_teacher_reply_and_request_tokens_retained':False,
            'later_teacher_reply_and_request_tokens_deliberately_excluded':True,
            'clarification_question_without_backend_context':True,
            'actual_private_receipt':None,'future_recorded_receipts':[],
            'question_only_selection':{'source_first_word_index':0,'source_end_word_index':1,'source_end_frame':78},
            'native_response_boundary_version':3,'future_first_real_frame':40,'source_start_frame':60,
            'future_end_real_frame':58,'question_deadline_s':5.,'question_retiming_before_project_verified':True}
        self.assertEqual(approved_words(m,side,actual)[0],[side['alignments'][0][0]])
        m['question_deadline_s']=4.8
        with self.assertRaises(RuntimeError):approved_words(m,side,actual)


if __name__=='__main__':unittest.main()
