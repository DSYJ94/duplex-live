import copy
import unittest
from evals.context_action_targets import cancel_window


class CancellationActionScope(unittest.TestCase):
    def scene(self):
        source={'split':'train','kind':'cancel'}
        teacher={'source_split':'train','kind':'cancel','source_model':'gpt-live-1',
            'request_policy':'native-prelead-epad','derived_requests':[
                {'operation':'lookup','derived_request_frame':5},
                {'operation':'cancel','derived_request_frame':20}]}
        side={'timeline':{'events':[{'ch':'user','kind':'utterance','text':'Cancel that lookup.',
                'actual_end_ms':1000}]},'user_words':[{'end_s':1.}],
            'alignments':[['Okay',[1.8,2.],'SPEAKER_MAIN']]}
        text=[3]*30;text[19:21]=[0,4]
        return source,teacher,side,text

    def test_only_approved_native_control_labels_are_selected(self):
        args=self.scene();before=copy.deepcopy(args)
        self.assertEqual(cancel_window(*args),(18,21));self.assertEqual(args,before)

    def test_heldout_wrong_kind_and_unapproved_teacher_are_rejected(self):
        for key,value in [('split','validation'),('kind','private')]:
            a,t,s,x=self.scene();a[key]=value
            with self.assertRaises(RuntimeError):cancel_window(a,t,s,x)
        for key,value in [('source_split','validation'),('kind','simple'),('source_model','student')]:
            a,t,s,x=self.scene();t[key]=value
            with self.assertRaises(RuntimeError):cancel_window(a,t,s,x)

    def test_incomplete_user_instruction_or_spoken_word_collision_rejected(self):
        a,t,s,x=self.scene();s['timeline']['events'][0]['actual_end_ms']=1500
        with self.assertRaises(RuntimeError):cancel_window(a,t,s,x)
        a,t,s,x=self.scene();s['user_words'][0]['end_s']=1.5
        with self.assertRaises(RuntimeError):cancel_window(a,t,s,x)
        a,t,s,x=self.scene();s['alignments'][0][1]=[1.5,1.7]
        with self.assertRaises(RuntimeError):cancel_window(a,t,s,x)

    def test_request_without_negative_cues_or_with_ambiguous_operations_rejected(self):
        a,t,s,x=self.scene();x[18]=99
        with self.assertRaises(RuntimeError):cancel_window(a,t,s,x)
        a,t,s,x=self.scene();t['derived_requests'][0]['operation']='cancel'
        with self.assertRaises(RuntimeError):cancel_window(a,t,s,x)


if __name__=='__main__':unittest.main()
