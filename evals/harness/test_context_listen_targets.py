import unittest
from evals.context_listen_targets import listen_window
from evals.context_action_component import planner


class ListenWindowTest(unittest.TestCase):
    def scene(self):
        return ({'split':'train','kind':'simple'},
            {'source_split':'train','kind':'simple','source_model':'gpt-live-1'},
            {'reference_arrivals':[],'timeline':{'events':[{'ch':'user','kind':'utterance'}]},
                'alignments':[['The answer.',[3.,4.],'SPEAKER_MAIN']],
                'user_words':[{'end_s':2.}]},[3]*160)

    def test_complete_measured_silence_has_fifty_control_targets(self):
        self.assertEqual(listen_window(*self.scene()),(55,105))

    def test_no_heldout_or_private_sources(self):
        for key,value in (('split','validation'),('kind','private'),('kind','cancel')):
            a,t,s,x=self.scene();a[key]=value
            with self.assertRaises(RuntimeError):listen_window(a,t,s,x)

    def test_no_later_words_requests_or_cropped_window(self):
        for token in (4,22,-1):
            a,t,s,x=self.scene();x[70]=token
            with self.assertRaises(RuntimeError):listen_window(a,t,s,x)
        a,t,s,x=self.scene()
        with self.assertRaises(RuntimeError):listen_window(a,t,s,x[:100])

    def test_receipt_or_future_user_speech_cannot_acquire_silence_labels(self):
        a,t,s,x=self.scene();s['reference_arrivals']=[{'ready_at_s':3.}]
        with self.assertRaises(RuntimeError):listen_window(a,t,s,x)
        a,t,s,x=self.scene();s['user_words'][0]['end_s']=5.
        with self.assertRaises(RuntimeError):listen_window(a,t,s,x)

    def test_unknown_action_kinds_fail_closed(self):
        with self.assertRaises(RuntimeError):planner('invented_action')
        self.assertEqual(planner('cancel_request')[1],'context_action_targets.py')


if __name__=='__main__':unittest.main()
