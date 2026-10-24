from copy import deepcopy
import json
import unittest
from evals.native_reasoning_timing_reapproval import changed_messages,IDS


class DerivedInputs(unittest.TestCase):
    def test_complete_payload_changes_only_diagnosed_recognition_fields(self):
        sent={f'native-reasoning4-train-{i:03d}':[
            dict(role='system',content='unchanged reviewer'),
            dict(role='user',content=json.dumps(dict(authored_user_utterances=['question'],
                actual_tasks_and_receipts={'task':'committed'},independent_asr_text='old',
                independently_measured_assistant_words=[dict(text='word',start=1,end=2)]),sort_keys=True))]
            for i in range(32)}
        before=deepcopy(sent)
        qualified=dict(results=[dict(id=k,asr_text='new',measured_words=[dict(text='word',start=3,end=4)])for k in sent])
        result=changed_messages(sent,qualified)
        self.assertEqual(sent,before)
        self.assertEqual(sum(result[k]!=sent[k]for k in sent),2)
        for k in IDS:
            self.assertEqual(result[k][0],sent[k][0])
            p=json.loads(result[k][1]['content'])
            self.assertEqual(p['actual_tasks_and_receipts'],{'task':'committed'})
            self.assertEqual(p['authored_user_utterances'],['question'])
            self.assertEqual(p['independently_measured_assistant_words'][0]['start'],3)


if __name__=='__main__':
    unittest.main()
