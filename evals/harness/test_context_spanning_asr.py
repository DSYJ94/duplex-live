import unittest
from evals.context_spanning_asr import context, request_context


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

    def test_unusable_recognition_becomes_an_explicit_no_execution_notice(self):
        result={'segments':[{'words':[{'text':'cancel','start':1.,'end':2.}]}]}
        rows,rejection=request_context(result,1.1,'',1.2,'ret-1')
        self.assertEqual(rows,[])
        self.assertEqual(rejection['route'],'error')
        self.assertEqual(rejection['receipts'],[])
        self.assertFalse(rejection['backend_api_called'])
        self.assertIn('repeat',rejection['content'])
        self.assertEqual(result['segments'][0]['words'][0]['end'],2.)

    def test_valid_words_remain_unchanged_and_unknown_failures_are_not_hidden(self):
        result={'segments':[{'words':[{'text':'cancel','start':.2,'end':.8}]}]}
        rows,rejection=request_context(result,1.1,'',1.2,'ret-1')
        self.assertIsNone(rejection)
        self.assertEqual(rows[0]['text'],'cancel')
        with self.assertRaises(KeyError):request_context({},1.1,'',1.2,'ret-1')


if __name__=='__main__':unittest.main()
