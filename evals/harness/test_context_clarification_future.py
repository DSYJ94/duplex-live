import unittest
from evals.context_clarification_future import question_scope


class QuestionScopeTests(unittest.TestCase):
    def scene(self):
        actual={'split':'train','kind':'clarify'}
        teacher={'source_split':'train','kind':'clarify','source_model':'gpt-live-1'}
        side={'timeline':{'meta':{'project':'Elm North','field':'locker label','phrase':'brown triangle'},
            'events':[{'ch':'user','kind':'utterance','text':"What's the locker label?",
                'actual_start_ms':800,'actual_end_ms':3200},
                {'ch':'user','kind':'utterance','text':'Project Elm North.',
                'actual_start_ms':6720,'actual_end_ms':9370}]},
            'alignments':[['Which project should I check for the locker label?', [3.96,6.08], 'SPEAKER_MAIN'],
                ['The label is brown triangle.',[20.,22.],'SPEAKER_MAIN']],
            'reference_arrivals':[{'ready_at_s':19.}]}
        return actual,teacher,side

    def test_complete_question_excludes_later_fact(self):
        self.assertEqual(question_scope(*self.scene()),(1,6.08))

    def test_validation_is_not_training(self):
        a,t,s=self.scene();a['split']='validation'
        with self.assertRaises(RuntimeError):question_scope(a,t,s)

    def test_future_project_is_not_already_known(self):
        a,t,s=self.scene();s['timeline']['events'][0]['text']='What is the locker label for Elm North?'
        with self.assertRaises(RuntimeError):question_scope(a,t,s)

    def test_question_cannot_overlap_project_reply(self):
        a,t,s=self.scene();s['timeline']['events'][1]['actual_start_ms']=6100
        with self.assertRaises(RuntimeError):question_scope(a,t,s)

    def test_question_cannot_contain_private_value(self):
        a,t,s=self.scene();s['alignments'][0][0]='Which project has the locker label brown triangle?'
        with self.assertRaises(RuntimeError):question_scope(a,t,s)

    def test_question_cannot_follow_an_answer_bearing_receipt(self):
        a,t,s=self.scene();s['reference_arrivals'][0]['ready_at_s']=5.
        with self.assertRaises(RuntimeError):question_scope(a,t,s)

    def test_missing_question_and_mismatched_field_fail(self):
        for replacement in ('Which project should I check?','Which project has the locker label.'):
            a,t,s=self.scene();s['alignments'][0][0]=replacement
            with self.assertRaises(RuntimeError):question_scope(a,t,s)


if __name__=='__main__':unittest.main()
