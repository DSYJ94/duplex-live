import unittest
from evals.context_request_prefix import scope


class LookupPrefixScopeTest(unittest.TestCase):
    def setUp(self):
        self.m={'teacher':{'kind':'private','source_split':'train','source_model':'gpt-live-1',
            'request_policy':'native-prelead-epad','derived_requests':[{'operation':'lookup','derived_request_frame':12}]},
            'actual_prefix_has_no_reply_or_request_tokens':True,'all_teacher_reply_and_request_tokens_retained':True,
            'source_start_frame':10,'future_first_real_frame':10,'future_real_frames':5,
            'real_to_future_virtual':list(range(5)),
            'calls':[{'phase':'live','route':'graph','start':i,'frames':1} for i in range(5)],
            'future_recorded_receipts':[{'real_frame':14}]}
        self.labels=[3,0,4,7,3];self.mask=[True]*5

    def check(self):return scope(self.m,self.labels,self.mask)

    def test_only_request_prefix_is_selected_and_later_words_and_receipts_are_excluded(self):
        self.assertEqual(self.check(),(3,3))

    def test_receipt_at_or_before_request_cannot_leak_into_lookup_decision(self):
        for frame in (11,12):
            with self.subTest(frame=frame):
                self.m['future_recorded_receipts']=[{'real_frame':frame}]
                with self.assertRaises(RuntimeError):self.check()

    def test_spoken_word_or_another_request_in_prefix_is_rejected(self):
        for token in (4,5,123):
            with self.subTest(token=token):
                self.labels[0]=token
                with self.assertRaises(RuntimeError):self.check()

    def test_missing_native_epad_or_target_is_rejected(self):
        self.labels[1]=3
        with self.assertRaises(RuntimeError):self.check()
        self.setUp();self.mask[2]=False
        with self.assertRaises(RuntimeError):self.check()

    def test_virtual_span_cannot_be_hidden_by_slicing(self):
        self.m['calls'][0]['phase']='span';self.m['calls'][0]['route']='direct'
        with self.assertRaises(RuntimeError):self.check()

    def test_heldout_or_nonprivate_or_unapproved_policy_cannot_supply_positive_labels(self):
        for key,value in (('source_split','validation'),('source_split','test'),('kind','simple'),
                          ('source_model','student'),('request_policy','source-dispatch')):
            with self.subTest(key=key,value=value):
                self.setUp();self.m['teacher'][key]=value
                with self.assertRaises(RuntimeError):self.check()

    def test_partial_future_or_prior_reply_cannot_enter_request_prefix(self):
        for key in ('all_teacher_reply_and_request_tokens_retained','actual_prefix_has_no_reply_or_request_tokens'):
            self.setUp();self.m[key]=False
            with self.assertRaises(RuntimeError):self.check()

    def test_cancellation_or_multiple_request_operations_are_rejected(self):
        self.m['teacher']['derived_requests'][0]['operation']='cancel'
        with self.assertRaises(RuntimeError):self.check()
        self.setUp();self.m['teacher']['derived_requests']*=2
        with self.assertRaises(RuntimeError):self.check()


if __name__=='__main__':unittest.main()
