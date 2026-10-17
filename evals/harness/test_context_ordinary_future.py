import copy
import unittest
from evals.context_ordinary_future import scope


class OrdinaryScopeTest(unittest.TestCase):
    def setUp(self):
        self.record = {'source':{'split':'train','kind':'simple','declared_counterfactual_world':{}}}
        self.report = {'source_split':'train','backend_execution_disabled':True,
            'conditioning_only':True,'training_approved':False,
            'native_foreground_backend_request':False,'worker_events':[]}
        self.actual = {'split':'train','kind':'simple'}
        self.teacher = {'kind':'simple','source_split':'train','source_model':'gpt-live-1'}
        self.side = {'reference_arrivals':[], 'timeline':{'events':[{'ch':'user','kind':'utterance'}]},
            'alignments':[['Four.',[4.8,5.0],'SPEAKER_MAIN']], 'user_words':[{'end_s':3.4}]}

    def check(self):
        return scope(self.record,self.report,[],self.actual,self.teacher,self.side)

    def test_clean_single_question_without_backend_is_a_candidate(self):
        self.assertIsNone(self.check())

    def test_heldout_or_private_source_cannot_use_ordinary_gate(self):
        for key,value in [('split','validation'),('split','test'),('kind','private'),('kind','cancel')]:
            with self.subTest(key=key,value=value):
                self.setUp();self.actual[key] = value
                with self.assertRaisesRegex(RuntimeError,'simple TRAIN'):self.check()

    def test_backend_answer_cannot_be_hidden_by_observation_metadata(self):
        self.report['worker_events'] = [{'event':'backend_result_ready','result':{'content':'The answer is four.'}}]
        with self.assertRaisesRegex(RuntimeError,'without backend'):self.check()
        self.setUp();self.record['source']['declared_counterfactual_world'] = {'secret':'four'}
        with self.assertRaisesRegex(RuntimeError,'without backend'):self.check()

    def test_teacher_receipt_or_request_cannot_leak_the_answer(self):
        self.side['reference_arrivals'] = [{'text':'The answer is four.'}]
        with self.assertRaisesRegex(RuntimeError,'without backend'):self.check()
        self.setUp();self.side['timeline']['events'].append({'ch':'agent','kind':'marker'})
        with self.assertRaisesRegex(RuntimeError,'without backend'):self.check()

    def test_future_user_words_or_second_question_are_rejected(self):
        self.side['user_words'][0]['end_s'] = 5.2
        with self.assertRaisesRegex(RuntimeError,'complete single'):self.check()
        self.setUp();self.side['timeline']['events'].append(copy.deepcopy(self.side['timeline']['events'][0]))
        with self.assertRaisesRegex(RuntimeError,'complete single'):self.check()

    def test_nonapproved_teacher_fold_is_rejected(self):
        self.teacher['source_split'] = 'validation'
        with self.assertRaisesRegex(RuntimeError,'approved TRAIN'):self.check()


if __name__ == '__main__':unittest.main()
