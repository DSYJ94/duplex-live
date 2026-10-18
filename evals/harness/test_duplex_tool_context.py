import json
import copy
import unittest
from evals.recovery_worker import RecordWorld,tool
from evals.duplex_tool_context import project_execution
from evals.qwen_response_executor import QwenResponseExecutor


class ActualControlHistory(unittest.TestCase):
    def source(self):
        world=RecordWorld({'Cedar':{'badge':'violet triangle'}});receipt=world.read('old','Cedar','badge')
        call=dict(type='function_call',call_id='actual-call',name='lookup_record',arguments=json.dumps(dict(project='Cedar',field='badge')))
        return call,receipt
    def test_actual_read_notice_withholds_value_and_rejects_mismatched_receipt(self):
        call,receipt=self.source();events=project_execution(call,receipt)
        self.assertNotIn(receipt['value'],json.dumps(events));self.assertTrue(json.loads(events[1]['text'])['value_withheld'])
        self.assertEqual(events[0]['tool_calls'][0]['id'],events[1]['tool_call_id'])
        for bad in (dict(receipt,project='Other'),dict(receipt,executed=False),dict(receipt,read_only=False)):
            with self.assertRaises(ValueError):project_execution(call,bad)
    def test_local_template_receives_real_control_roles_and_rejects_orphan_notice(self):
        call,receipt=self.source();events=project_execution(call,receipt);captured=[]
        class Generator:
            def request(self,messages,tools,assistant_prefill):
                captured.append(copy.deepcopy(messages))
                return dict(cleaned_text='<tool_call>{"name":"respond_to_user","arguments":{"route":"answer","content":"Three."}}</tool_call>',input_tokens=1,generated_tokens=1)
        executor=object.__new__(QwenResponseExecutor);executor.model_id='local';executor.prompt_profile='base';executor.pending_speech_protocol='array';executor.counter=0;executor.history={};executor.generator=Generator();executor.attention_backend='math'
        payload=dict(model='local',local_worker_state=dict(available=dict(records={},tasks={}),pending_speech=[],dispatch_protocol='required-action'),
            tool_choice='required',tools=[tool('respond_to_user','Ordinary reply',dict(route=dict(type='string',enum=['answer','clarify']),content=dict(type='string')))],
            input=json.dumps([dict(role='user',text='Check Cedar badge.')]+events+[dict(role='user',text='How many triangle sides?')]))
        result=executor._request(payload)
        self.assertEqual([m['role'] for m in captured[0]],['system','user','assistant','tool','user'])
        self.assertEqual(captured[0][2]['tool_calls'][0]['id'],'actual-call');self.assertNotIn('violet triangle',json.dumps(captured))
        payload['input']=json.dumps([events[1],dict(role='user',text='How many triangle sides?')])
        with self.assertRaisesRegex(RuntimeError,'matching chronological call'):executor._request(payload)


if __name__=='__main__':unittest.main()
