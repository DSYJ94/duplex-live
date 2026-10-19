import copy
import unittest
from evals.recovery_teacher_convert import measured_question_completion, measured_timeline


class MeasuredQuestionCompletion(unittest.TestCase):
    def fixture(self):
        source=dict(meta=dict(kind='private'),events=[dict(ch='user',kind='utterance',text='Check Laurel South.',
                    start_ms=800,actual_start_ms=800,actual_end_ms=6900)])
        words=[dict(text='Check',start_s=.8,end_s=1.2),dict(text='Laurel',start_s=2.,end_s=3.5),
               dict(text='South.',start_s=3.5,end_s=3.8)]
        return source,words

    def test_full_measured_question_is_distinct_from_unchanged_wave_file_tail(self):
        source,words=self.fixture();proof=measured_question_completion(source,words)
        self.assertEqual(proof['complete_at_s'],3.8)
        self.assertEqual(proof['questions'][0]['enclosing_clip_end_s'],6.9)
        asst=[dict(text='Checking.',start=7.,end=7.4)]
        starts=[dict(received_at_s=5.5,delegation=dict(id='lookup'))]
        with self.assertRaisesRegex(ValueError,'predates complete'):
            measured_timeline('legacy',asst,source,starts,[])
        timeline,error=measured_timeline('new-version',asst,source,starts,[],user_words=words)
        self.assertLessEqual(error,200)
        self.assertEqual(timeline.meta['question_completion_clock'],proof)
        self.assertEqual(timeline.events[0].actual_end_ms,6900)
        early=[dict(received_at_s=3.7,delegation=dict(id='lookup'))]
        with self.assertRaisesRegex(ValueError,'predates complete'):
            measured_timeline('reject',asst,source,early,[],user_words=words)

    def test_missing_reordered_or_outside_words_cannot_complete_a_question(self):
        source,words=self.fixture()
        for changed in (words[:-1],list(reversed(words)),[dict(w,end_s=8.) for w in words]):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                measured_question_completion(source,changed)

    def test_clarification_requires_complete_measured_project_identification(self):
        source,words=self.fixture();source=copy.deepcopy(source);source['meta']['kind']='clarify'
        source['events'].append(dict(ch='user',kind='utterance',text='Project Laurel South.',
            start_ms=8000,actual_start_ms=8000,actual_end_ms=10000))
        with self.assertRaises(ValueError):measured_question_completion(source,words)
        words=words+[dict(text='Project',start_s=8.,end_s=8.2),dict(text='Laurel',start_s=8.2,end_s=8.6),
                     dict(text='South.',start_s=8.6,end_s=9.)]
        self.assertEqual(measured_question_completion(source,words)['complete_at_s'],9.)


if __name__=='__main__':unittest.main()
