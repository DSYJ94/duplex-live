import unittest
import json
import types
from unittest.mock import patch
from evals.recovery_worker import RecordWorld,WorkerSuperseded,WorkerContractError,run_worker,lookup_receipts


class AtomicLookupReplacement(unittest.TestCase):
    def world(self):
        world=RecordWorld({'Oak Central':{'meeting room':'room seven'},'Pine Central':{'meeting room':'room twelve'}})
        old=world.read('old','Oak Central','meeting room');other=world.read('other','Oak Central','meeting room')
        revision=world.reserve('new')
        return world,old,other,revision

    def test_actual_controller_replacement_preserves_nested_execution_and_delivery_guards(self):
        world,old,other,revision=self.world()
        responses=[{'id':'replace','status':'completed','output':[{'type':'function_call','name':'replace_lookup','call_id':'replace',
            'arguments':json.dumps({'task_ids':['old'],'project':'Pine Central','field':'meeting room'})}]},
            {'id':'reply','status':'completed','output':[{'type':'message','content':[{'type':'output_text',
             'text':json.dumps({'route':'answer','content':'Pine Central uses room twelve.','consume_pending_speech_ids':[]})}]}]}]
        def post(*args,**kwargs):return types.SimpleNamespace(status_code=200,json=lambda:responses.pop(0))
        with patch.dict('sys.modules',{'requests':types.SimpleNamespace(post=post)}),patch.dict('os.environ',{'OPENAI_API_KEY':'test-placeholder'}):
            result=run_worker([{'role':'user','text':'Actually use Project Pine Central instead.'}],world,'new',
                expected_revision=revision,dispatch_protocol='state-aware')
        reads=lookup_receipts(result['receipts']);self.assertEqual(result['route'],'delegate')
        self.assertEqual(len(reads),1);self.assertEqual(reads[0]['value'],'room twelve')
        self.assertFalse(world.deliverable(old));self.assertTrue(world.deliverable(other));self.assertTrue(world.deliverable(reads[0]))
        world.cancel('new');self.assertFalse(world.deliverable(reads[0]))

    def test_unidentified_replacement_cannot_cancel_existing_work(self):
        world,old,other,revision=self.world()
        def post(*args,**kwargs):return types.SimpleNamespace(status_code=200,json=lambda:{'id':'bad','status':'completed','output':[
            {'type':'function_call','name':'replace_lookup','call_id':'bad',
             'arguments':json.dumps({'task_ids':['old'],'project':'Pine Central','field':'meeting room'})}]})
        with patch.dict('sys.modules',{'requests':types.SimpleNamespace(post=post)}),patch.dict('os.environ',{'OPENAI_API_KEY':'test-placeholder'}):
            with self.assertRaises(WorkerContractError):
                run_worker([{'role':'user','text':'Use the other project instead.'}],world,'new',expected_revision=revision,dispatch_protocol='state-aware')
        self.assertTrue(world.deliverable(old));self.assertTrue(world.deliverable(other));self.assertEqual(len(world.receipts),2)

    def test_replace_executes_new_read_and_cancels_only_selected_old_work(self):
        world,old,other,revision=self.world()
        result=world.replace_lookup('new',['old'],'Pine Central','meeting room',revision)
        self.assertFalse(world.deliverable(old));self.assertTrue(world.deliverable(other))
        self.assertTrue(world.deliverable(result['lookup']));self.assertEqual(result['lookup']['value'],'room twelve')
        self.assertEqual(result['cancellation']['task_ids'],['old']);self.assertTrue(result['executed'])

    def test_bad_new_field_or_revision_cannot_cancel_existing_work(self):
        for field,revision_delta in [('unknown',0),('meeting room',1)]:
            world,old,other,revision=self.world()
            with self.assertRaises((ValueError,WorkerSuperseded)):
                world.replace_lookup('new',['old'],'Pine Central',field,revision+revision_delta)
            self.assertTrue(world.deliverable(old));self.assertTrue(world.deliverable(other))
            self.assertEqual(len(world.receipts),2)

    def test_bad_old_targets_cannot_execute_new_read_or_touch_valid_work(self):
        for targets in [['old','unknown'],['old','old'],['new'],[]]:
            world,old,other,revision=self.world()
            with self.assertRaises(ValueError):world.replace_lookup('new',targets,'Pine Central','meeting room',revision)
            self.assertTrue(world.deliverable(old));self.assertTrue(world.deliverable(other))
            self.assertEqual(len(world.receipts),2)


if __name__=='__main__':unittest.main()
