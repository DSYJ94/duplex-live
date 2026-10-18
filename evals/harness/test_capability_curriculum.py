import json
import unittest
from training.control.capability_curriculum import specifications,run,assess
from training.control.capability_data import api_history
from training.control.curriculum import xml_action


class ReceiptStageBoundary(unittest.TestCase):
    def test_private_value_enters_student_only_after_real_fixture_read(self):
        scene=next(s for s in specifications(True) if s['split']=='train' and s['kind']=='record-read');value=scene['expected']['exact_content'];calls=[]
        class Fixture:
            def request(self,messages,tools,assistant_prefill=''):
                calls.append((messages,tools));index=len(calls)-1
                if index==0:
                    self.assert_missing=value not in json.dumps(messages)
                    action=dict(name='lookup_record',arguments=dict(project=scene['expected']['project'],field=scene['expected']['field']))
                else:
                    self.assert_visible=any(m.get('role')=='tool' and value in m.get('content','') for m in messages);action=dict(name='answer',arguments=dict(content=value))
                text=xml_action(action);return dict(raw_text=text,cleaned_text=text,assistant_prefill=assistant_prefill,input_tokens=0,generated_tokens=0)
        generator=Fixture();result,world=run(scene,generator,'fixture-only');self.assertTrue(assess(scene,result,world)['passed'])
        self.assertTrue(generator.assert_missing);self.assertTrue(generator.assert_visible);self.assertEqual(len(world.receipts),1)
        converted=api_history(calls[1][0]);prior=next(x for x in converted if x.get('type')=='function_call');receipt=next(x for x in converted if x.get('type')=='function_call_output')
        self.assertEqual(prior['name'],'lookup_record');self.assertIsInstance(prior['arguments'],str)
        self.assertEqual(receipt['type'],'function_call_output');self.assertEqual(receipt['call_id'],prior['call_id']);self.assertEqual(json.loads(receipt['output'])['value'],value)
    def test_completed_result_never_acquires_cancellation_receipt_when_resuming(self):
        scene=next(s for s in specifications(True) if s['split']=='train' and s['kind']=='finished-resume')
        class Fixture:
            def request(self,messages,tools,assistant_prefill=''):
                names={t['function']['name'] for t in tools}
                if 'cancel_reasoning' in names or 'cancel_reasoning_and_resume_speech' in names:raise AssertionError('Closed task incorrectly cancellable')
                text=xml_action(dict(name='resume_speech',arguments={}))
                return dict(raw_text=text,cleaned_text=text,assistant_prefill=assistant_prefill,input_tokens=0,generated_tokens=0)
        result,world=run(scene,Fixture(),'fixture-only');self.assertTrue(assess(scene,result,world)['passed']);self.assertEqual(result['receipts'],[])


if __name__=='__main__':unittest.main()
