import unittest
from evals.lychee_semantic_layout import continuation


class NativeLayoutTests(unittest.TestCase):
    def test_future_words_follow_receipt_and_native_control_boundary(self):
        config={'text_pad_token_id':1,'stoken_pad_token_id':2,'sleep_token_id':3,'detect_token_id':4,
            'start_speaking_token_id':5,'start_listening_token_id':6,'keep_speaking_token_id':7,'keep_listening_token_id':13,
            'stoken_delay_token_id':8,'stoken_delay_num':10}
        tokens={'eot':9,'tts_start':10,'tts_end':11,'tts_pad':12}
        r=continuation(760,[100,101],[1,2,3,4],config,tokens,771)
        self.assertGreaterEqual(r['first_word_slot'],771)
        self.assertEqual(r['ss_slot']%10,9)
        self.assertEqual(r['control_ids'][r['ss_slot']-760],5)
        self.assertEqual(r['control_ids'][9],13)
        self.assertEqual([r['text_ids'][i-760] for i in r['supervised_slots']],[100,101,9])
        sound=[v for v in r['stoken_ids'] if v>=151696]
        self.assertEqual(sound,[151697,151698,151699,151700])
        self.assertEqual(r['control_ids'][-1],6)


if __name__=='__main__':unittest.main()
