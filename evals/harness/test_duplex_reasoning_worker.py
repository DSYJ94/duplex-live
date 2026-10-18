import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from evals.duplex_reasoning_worker import ReasoningWorld,reasoning_receipts,run_reasoning_worker


class ReasoningBoundary(unittest.TestCase):
    def test_actual_request_does_not_block_foreground_reservation_and_cancel_hides_result(self):
        started=threading.Event();release=threading.Event();received=[]
        def backend(question):
            received.append(question);started.set();self.assertTrue(release.wait(2));return dict(content='Nineteen.',model='test-only',response_id='fixture',remote_api_used=True)
        world=ReasoningWorld({},backend);revision=world.reserve('old')
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(world.reason,'old',revision,'Exact user question')
            try:
                self.assertTrue(started.wait(1));world.reserve('new');world.cancel('old')
            finally:release.set()
            receipt=future.result()
        self.assertEqual(received,['Exact user question']);self.assertFalse(world.deliverable(receipt));self.assertEqual(reasoning_receipts([receipt]),[receipt])
    def test_model_selects_action_but_application_binds_original_request_and_preserves_receipt(self):
        received=[]
        def backend(question):
            received.append(question);return dict(content='Nineteen.',model='test-only',response_id='fixture',remote_api_used=True)
        world=ReasoningWorld({},backend);revision=world.reserve('current');context=[dict(role='user',text='Actual counting question')]
        class Controller:
            provider_name='fixture-only'
            def __call__(self,payload):
                self.payload=payload
                return dict(id='fixture',status='completed',model='fixture-only',output=[dict(type='function_call',name='delegate_reasoning',arguments='{}')])
        controller=Controller();result=run_reasoning_worker(context,world,'current','fixture-only',revision,'state-aware',[],controller)
        self.assertEqual(received,['Actual counting question']);self.assertEqual(result['content'],'Nineteen.');self.assertEqual(result['route'],'delegate')
        self.assertEqual(controller.payload['local_worker_state']['domain'],'reasoning-development');self.assertEqual(result['receipts'][0],world.receipts[0])
        self.assertFalse(result['cancellation_supported'])
    def test_private_records_and_fabricated_receipts_reject(self):
        with self.assertRaises(ValueError):ReasoningWorld({'Private':{'field':'value'}})
        for receipt in ({'tool':'lookup_record'},{'tool':'delegate_reasoning','executed':True,'content':'Claim'}):
            with self.assertRaises(ValueError):reasoning_receipts([receipt])


if __name__=='__main__':unittest.main()
