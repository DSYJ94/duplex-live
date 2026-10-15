import unittest
from evals.context_spanning_asr import context


class ArrivedAudioContextTest(unittest.TestCase):
    def test_only_actual_asr_bounds_define_user_utterances(self):
        result={'segments':[{'words':[{'text':'Willow','start':.2,'end':.4},
            {'text':'North','start':.5,'end':.7},{'text':'why','start':2.,'end':2.2}]}]}
        rows=context(result,2.3,'I am checking.',2.4)
        self.assertEqual([r['text'] for r in rows],['Willow North','why','I am checking.'])
        self.assertEqual(rows[0]['end_s'],.7)
        self.assertNotIn('start_s',rows[-1])

    def test_future_or_missing_recognition_cannot_use_reference_fallback(self):
        for result in ({'segments':[]},{'segments':[{'words':[{'text':'future','start':1.,'end':2.}]}]}):
            with self.assertRaises(RuntimeError):context(result,1.1,'',1.2)


if __name__=='__main__':unittest.main()
