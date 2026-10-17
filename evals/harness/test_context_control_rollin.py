import unittest
from evals.context_control_rollin import lookup_window
from evals.harness.test_context_clarification_future import QuestionScopeTests


class ClarifiedLookupWindowTest(unittest.TestCase):
    def scene(self):
        source,teacher,side=QuestionScopeTests().scene()
        teacher.update(request_policy='native-prelead-epad',derived_requests=[dict(operation='lookup',derived_request_frame=135)])
        text=[3]*200;text[134]=0;text[135]=4
        return source,teacher,side,text

    def test_lookup_follows_the_complete_identified_project(self):
        self.assertEqual(lookup_window(*self.scene()),(133,136))

    def test_partial_project_and_future_question_are_not_authorized(self):
        source,teacher,side,text=self.scene();side['timeline']['events'][1]['actual_end_ms']=10800
        with self.assertRaises(RuntimeError):lookup_window(source,teacher,side,text)
        source,teacher,side,text=self.scene()
        side['timeline']['events'].append(dict(ch='user',kind='utterance',actual_start_ms=11000,actual_end_ms=14000,text='Another question.'))
        with self.assertRaises(RuntimeError):lookup_window(source,teacher,side,text)

    def test_nonexpert_validation_wrong_action_and_spoken_words_rejected(self):
        for index,key,value in [(0,'split','validation'),(1,'source_model','student'),
                (1,'derived_requests',[dict(operation='cancel',derived_request_frame=135)])]:
            args=list(self.scene());args[index][key]=value
            with self.assertRaises(RuntimeError):lookup_window(*args)
        args=list(self.scene());args[3][134]=7
        with self.assertRaises(RuntimeError):lookup_window(*args)
        args=list(self.scene());args[2]['alignments'].append(['Unexpected word.',[10.7,10.9],'SPEAKER_MAIN'])
        with self.assertRaises(RuntimeError):lookup_window(*args)


if __name__=='__main__':unittest.main()
