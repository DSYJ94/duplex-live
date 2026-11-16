"""Protocol integration with fixture generation, never native-model success."""
import unittest,threading,types
from unittest.mock import patch,Mock
from portable.context.local_reasoner_worker import Local27BExecutor
from evals.recovery_worker import RecordWorld,WorkerContractError


class FixtureExecutor(Local27BExecutor):
    def __init__(self,outputs):
        self.model_id='fixture-not-native';self.prompt_profile='base';self.pending_speech_protocol='array';self.attention_backend='math';self.spoken_output_contract=True;self.compact_replies=False
        self.lock=threading.RLock();self.history={};self.counter=0;self.outputs=iter(outputs);self.messages=[];self.generator=self
    def request(self,messages,tools,**kwargs):
        self.messages.append(messages)
        return dict(cleaned_text=next(self.outputs),input_tokens=10,generated_tokens=10,wall_s=.01)
    def __call__(self,payload):return self._request(payload)


def call(name,**args):
    import json
    def value(x):return x if isinstance(x,str)else json.dumps(x)
    return '<tool_call><function='+name+'>'+''.join('<parameter='+k+'>'+value(v)+'</parameter>'for k,v in args.items())+'</function></tool_call>'


class LocalWorkerTest(unittest.TestCase):
    def test_real_private_read_and_receipt_followup_use_existing_world_guards(self):
        executor=FixtureExecutor([call('lookup_record',project='Juniper',field='label'),call('respond_to_user',route='delegate',content='The label is amber square.',consume_pending_speech_ids=[])])
        world=RecordWorld({'Juniper':{'label':'amber square'}})
        with patch.dict('sys.modules',{'requests':types.SimpleNamespace(post=Mock(side_effect=AssertionError('No network')))}):
            result=executor.worker([dict(role='user',text='Please check the label for Project Juniper.')],world,'now',dispatch_protocol='state-aware',scope_read_tools=True)
        self.assertEqual(result['route'],'delegate');self.assertEqual(result['receipts'],world.receipts);self.assertTrue(world.receipts[0]['executed'])
        self.assertTrue(any(m['role']=='tool'and 'amber square'in m['content']for m in executor.messages[1]))
        self.assertEqual(result['response_provider'],'local-pinned-qwen36-27b')

    def test_unidentified_private_project_cannot_be_guessed_from_available_names(self):
        executor=FixtureExecutor([call('lookup_record',project='Juniper',field='label')]);world=RecordWorld({'Juniper':{'label':'amber square'}})
        with self.assertRaises(WorkerContractError):executor.worker([dict(role='user',text='Please check the label.')],world,'now',dispatch_protocol='state-aware',scope_read_tools=True)
        self.assertFalse(world.receipts)

    def test_ordinary_response_creates_no_tool_receipt(self):
        executor=FixtureExecutor([call('respond_to_user',route='answer',content='A triangle has three sides.',consume_pending_speech_ids=[])]);world=RecordWorld({})
        result=executor.worker([dict(role='user',text='How many sides does a triangle have?')],world,'now',dispatch_protocol='state-aware',scope_read_tools=True)
        self.assertEqual(result['route'],'answer');self.assertFalse(result['receipts']);self.assertFalse(world.receipts)
