import copy
import unittest
from evals.context_cancel_ack import completed_receipt,teacher_scope


class CancelAckTest(unittest.TestCase):
    def events(self):
        return [dict(event='backend_result_ready',applied_at_s=10.,result=dict(route='delegate',receipts=[
            dict(tool='lookup_record',task_id='ret-0',executed=True)])),
            dict(event='backend_result_ready',applied_at_s=22.,disposition='actual backend text injected',
                result=dict(route='cancel',receipts=[dict(tool='cancel_lookup',cancelled=True,task_ids=['ret-0'],
                    receipts=[dict(tool='cancel_lookup',cancelled=True,task_id='ret-0')])]))]

    def test_actual_completed_cancellation_is_required(self):
        events=self.events();self.assertIs(completed_receipt(events),events[1])
        for key,value in [('route','error'),('route','answer')]:
            bad=copy.deepcopy(events);bad[1]['result'][key]=value
            with self.assertRaises(RuntimeError):completed_receipt(bad)
        for key,value in [('applied_at_s',None),('applied_at_s',float('nan')),('disposition','rejected; cancelled task')]:
            bad=copy.deepcopy(events);bad[1][key]=value
            with self.assertRaises(RuntimeError):completed_receipt(bad)

    def test_false_or_mismatched_or_unexecuted_receipts_do_not_justify_words(self):
        for key,value in [('cancelled',False),('task_ids',['invented']),('receipts',[]),('task_ids',[])]:
            bad=self.events();bad[1]['result']['receipts'][0][key]=value
            with self.assertRaises(RuntimeError):completed_receipt(bad)
        bad=self.events();bad[0]['result']['receipts'][0]['executed']=False
        with self.assertRaises(RuntimeError):completed_receipt(bad)
        bad=self.events();bad[1]['result']['receipts'][0]['receipts'][0]['cancelled']=False
        with self.assertRaises(RuntimeError):completed_receipt(bad)
        bad=self.events();bad.append(copy.deepcopy(bad[-1]))
        with self.assertRaises(RuntimeError):completed_receipt(bad)
        bad=self.events();bad[0]['applied_at_s']=23.
        with self.assertRaises(RuntimeError):completed_receipt(bad)

    def scope_args(self):
        return [dict(split='train',kind='cancel'),dict(source=dict(split='train',kind='cancel')),
            dict(kind='cancel',source_split='train',source_model='gpt-live-1',request_policy='native-prelead-epad'),
            dict(reference_arrivals=[dict(ready_at_s=21.6,text='The lookup has been canceled.')],
                alignments=[["It's",[22.76,23.0],'SPEAKER_MAIN'],['cancelled.',[23.,23.32],'SPEAKER_MAIN']])]

    def test_only_independent_train_completion_words_after_teacher_receipt(self):
        teacher_scope(*self.scope_args())
        for index,key,value in [(0,'split','validation'),(1,'source',dict(split='train',kind='simple')),
                (2,'source_model','student'),(2,'source_split','test')]:
            args=self.scope_args();args[index][key]=value
            with self.assertRaises(RuntimeError):teacher_scope(*args)
        for words in [["I'll",'cancel.'],['Not','cancelled.']]:
            args=self.scope_args()
            for a,w in zip(args[3]['alignments'],words):a[0]=w
            with self.assertRaises(RuntimeError):teacher_scope(*args)
        args=self.scope_args();args[3]['reference_arrivals'][0]['ready_at_s']=22.7
        with self.assertRaises(RuntimeError):teacher_scope(*args)


if __name__=='__main__':unittest.main()
