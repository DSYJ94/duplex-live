import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from evals.duplex_reasoning_worker import ReasoningWorld,reasoning_receipts,run_reasoning_worker
from evals.recovery_worker import WorkerContractError


class ReasoningBoundary(unittest.TestCase):
    def test_actual_request_does_not_block_foreground_reservation_and_cancel_hides_result(self):
        started=threading.Event();release=threading.Event();received=[]
        def backend(question):
            received.append(question);started.set();self.assertTrue(release.wait(2));return dict(content='Nineteen.',model='test-only',response_id='fixture',remote_api_used=True)
        world=ReasoningWorld({},backend);revision=world.reserve('old')
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(world.reason,'old',revision,'Exact user question')
            try:
                self.assertTrue(started.wait(1));world.reserve('new');cancellation=world.cancel_delivery(['old']);self.assertTrue(cancellation['delivery_cancelled']);self.assertFalse(cancellation['remote_execution_aborted'])
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
        self.assertEqual(result['cancellation_supported'],'delivery-only');self.assertFalse(result['remote_execution_abort_supported'])
    def test_cancellation_validates_all_targets_atomically_and_preserves_unrelated_work(self):
        world=ReasoningWorld({});world.reserve('old');world.reserve('unrelated')
        world.tasks['old'].update(operation='delegate_reasoning',phase='running')
        with self.assertRaises(ValueError):world.cancel_delivery(['old','unrelated'])
        self.assertFalse(world.tasks['old']['cancelled'])
        world.cancel_delivery(['old']);self.assertTrue(world.tasks['old']['cancelled']);self.assertFalse(world.tasks['unrelated']['cancelled'])
    def test_private_records_and_fabricated_receipts_reject(self):
        with self.assertRaises(ValueError):ReasoningWorld({'Private':{'field':'value'}})
        for receipt in ({'tool':'lookup_record'},{'tool':'delegate_reasoning','executed':True,'content':'Claim'}):
            with self.assertRaises(ValueError):reasoning_receipts([receipt])

    def test_invalid_speech_cannot_cancel_real_background_work(self):
        class Controller:
            provider_name='fixture-only'
            def __init__(self,arguments):self.arguments=arguments
            def __call__(self,payload):
                return dict(id='fixture',status='completed',model='fixture-only',output=[dict(type='function_call',name='cancel_reasoning',arguments=json.dumps(self.arguments))])
        for content,consumed in [('Cancelled.',['unknown']),('   ',[])]:
            with self.subTest(content=content,consumed=consumed):
                world=ReasoningWorld({});world.reserve('old');world.tasks['old'].update(operation='delegate_reasoning',phase='running')
                revision=world.reserve('current');controller=Controller(dict(task_ids=['old'],content=content,consume_pending_speech_ids=consumed))
                with self.assertRaises(WorkerContractError):
                    run_reasoning_worker([dict(role='user',text='Cancel the old task')],world,'current','fixture-only',revision,'state-aware',[],controller)
                self.assertFalse(world.tasks['old']['cancelled'])

    def test_model_can_withdraw_known_speech_without_new_words_or_remote_operation(self):
        class Controller:
            provider_name='fixture-only'
            def __call__(self,payload):
                self.payload=payload
                return dict(id='fixture',status='completed',model='fixture-only',output=[dict(type='function_call',name='withdraw_speech',arguments=json.dumps(dict(speech_ids=['old-answer'])))])
        world=ReasoningWorld({});world.reserve('background');world.tasks['background'].update(operation='delegate_reasoning',phase='running')
        revision=world.reserve('stop');controller=Controller()
        result=run_reasoning_worker([dict(role='user',text='Stop.')],world,'stop','fixture-only',revision,'state-aware',
            [dict(id='old-answer',request_text='Explain the Moon',text='Actual queued explanation')],controller)
        self.assertEqual(result['route'],'silence');self.assertEqual(result['content'],'')
        self.assertEqual(result['consume_pending_speech_ids'],['old-answer']);self.assertEqual(world.receipts,[])
        self.assertIn('withdraw_speech',[t['name'] for t in controller.payload['tools']])
        self.assertFalse(world.tasks['background']['cancelled'])

    def test_silent_withdrawal_rejects_unknown_duplicate_or_empty_targets(self):
        class Controller:
            provider_name='fixture-only'
            def __init__(self,arguments):self.arguments=arguments
            def __call__(self,payload):
                return dict(id='fixture',status='completed',model='fixture-only',output=[dict(type='function_call',name='withdraw_speech',arguments=json.dumps(self.arguments))])
        for arguments in [dict(speech_ids=[]),dict(speech_ids=['unknown']),dict(speech_ids=['known','known']),dict(speech_ids=['known'],content='Invented reply')]:
            with self.subTest(arguments=arguments):
                world=ReasoningWorld({});revision=world.reserve('current')
                with self.assertRaises(WorkerContractError):
                    run_reasoning_worker([dict(role='user',text='Stop.')],world,'current','fixture-only',revision,'state-aware',
                        [dict(id='known',request_text='Explain',text='An actual owed response')],Controller(arguments))
                self.assertEqual(world.receipts,[])

    def test_generated_history_forwards_delivery_uncertainty_to_the_model(self):
        from evals.duplex_speech_context import started,retired
        context=[dict(role='user',text='Explain a topic')]
        started(context,dict(task_id='old',result=dict(content='An actual generated reply')),'speech-1')
        retired(context,['old']);context.append(dict(role='user',text='Did that finish?'))
        class Controller:
            provider_name='fixture-only'
            def __call__(self,payload):
                self.payload=payload
                return dict(id='fixture',status='completed',model='fixture-only',output=[dict(type='function_call',name='respond_to_user',arguments=json.dumps(dict(route='answer',content='Fixture reply.',consume_pending_speech_ids=[])))])
        world=ReasoningWorld({});revision=world.reserve('current');controller=Controller()
        run_reasoning_worker(context,world,'current','fixture-only',revision,'state-aware',[],controller)
        self.assertEqual(controller.payload['local_worker_state']['generated_speech_delivery'],[
            dict(task_id='old',delivery='retired-before-full-playout',physical_hearing_confirmed=False)])


if __name__=='__main__':unittest.main()
