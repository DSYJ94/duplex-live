import io
import json
import threading
import types
import unittest
from unittest.mock import patch
from evals.qwen35_rpc import send,receive,ResidentQwen35,LIMIT
from evals.recovery_worker import WorkerContractError


class Channel:
    def __init__(self,reply):self.reply=(json.dumps(reply)+'\n').encode();self.written=b''
    def write(self,value):self.written+=value
    def flush(self):pass
    def readline(self,limit):return self.reply


class ResidentTransport(unittest.TestCase):
    def test_roundtrip_and_reject_incomplete_oversized_nonobject_nonfinite(self):
        stream=io.BytesIO();send(stream,dict(text='Actual words.'));stream.seek(0);self.assertEqual(receive(stream),dict(text='Actual words.'))
        for value in (b'',b'{"text":"cut"}',b'[]\n',b'x'*(LIMIT+1)):
            with self.subTest(value=value[:20]),self.assertRaises((ValueError,RuntimeError)):receive(io.BytesIO(value))
        with self.assertRaises(ValueError):send(io.BytesIO(),dict(value=float('nan')))
    def proxy(self,reply):
        proxy=object.__new__(ResidentQwen35);proxy.lock=threading.RLock();proxy.counter=0;proxy.parent_version='4.55.3'
        proxy.process=types.SimpleNamespace(poll=lambda:None);proxy.stream=Channel(reply);proxy.log_path='full-controller.log';return proxy
    def test_identity_and_failure_provenance_cannot_become_a_successful_action(self):
        with self.assertRaisesRegex(RuntimeError,'identity mismatch'):
            self.proxy(dict(kind='response',id=2,response={}))(dict(model='actual'))
        detail=dict(local_generation=dict(raw_text='Malformed actual output'))
        with self.assertRaises(WorkerContractError) as error:
            self.proxy(dict(kind='error',id=1,error_type='WorkerContractError',message='Rejected',result=detail))(dict(model='actual'))
        self.assertEqual(error.exception.result,detail)
    def test_actual_response_is_unchanged_and_parent_runtime_is_checked(self):
        response=dict(id='actual-local-response',status='completed',model='Qwen/Qwen3.5-9B',output=[])
        proxy=self.proxy(dict(kind='response',id=1,response=response))
        with patch.dict('sys.modules',{'transformers':types.SimpleNamespace(__version__='4.55.3')}):self.assertEqual(proxy(dict(model='actual')),response)
        proxy=self.proxy(dict(kind='response',id=1,response=response))
        with patch.dict('sys.modules',{'transformers':types.SimpleNamespace(__version__='5.5.3')}),self.assertRaisesRegex(RuntimeError,'runtime changed'):proxy(dict(model='actual'))


if __name__=='__main__':unittest.main()
