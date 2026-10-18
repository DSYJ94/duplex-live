import unittest
import json
import types
from unittest.mock import patch
from evals.recovery_worker import RecordWorld,run_worker,WorkerSuperseded,WorkerContractError,project_identified_by_user


class RecordGuards(unittest.TestCase):
    def test_execution_observer_reports_real_read_before_model_formats_its_receipt(self):
        world=RecordWorld({'Cedar':{'badge':'violet triangle'}});order=[];observed=[]
        def execute(payload):
            if 'previous_response_id' not in payload:
                order.append('action')
                return dict(id='read',status='completed',output=[dict(type='function_call',name='lookup_record',call_id='real-call',arguments=json.dumps(dict(project='Cedar',field='badge')))])
            order.append('reply')
            import threading
            other=threading.Thread(target=world.reserve,args=('other',));other.start();other.join(.2)
            self.assertFalse(other.is_alive(),'World lock must not be held while a model formats its reply')
            return dict(id='reply',status='completed',output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(dict(route='answer',content='The badge is violet triangle.')))])])
        def observe(event):order.append('executed');observed.append(event)
        result=run_worker([dict(role='user',text='Read the badge for Project Cedar.')],world,'current',model='local',
            dispatch_protocol='required-action',response_executor=execute,execution_observer=observe)
        self.assertEqual(order,['action','executed','reply']);self.assertEqual(observed[0]['receipt'],result['receipts'][0])
        self.assertEqual(observed[0]['call']['call_id'],'real-call');self.assertTrue(observed[0]['receipt']['executed'])

    def test_rejected_project_never_creates_an_execution_notice(self):
        world=RecordWorld({'Cedar':{'badge':'violet triangle'}});observed=[]
        def execute(payload):return dict(id='bad',status='completed',output=[dict(type='function_call',name='lookup_record',call_id='bad-call',arguments=json.dumps(dict(project='Cedar',field='badge')))])
        with self.assertRaises(WorkerContractError):
            run_worker([dict(role='user',text='What is the badge?')],world,'current',model='local',dispatch_protocol='required-action',
                response_executor=execute,execution_observer=observed.append)
        self.assertEqual(observed,[]);self.assertEqual(world.receipts,[])

    def test_scoped_read_tools_cannot_offer_unavailable_or_unidentified_projects(self):
        for records,user,expected in [({},'Read the access code for Project Missing.',set()),
                ({'Cedar':{'badge':'violet triangle'}},'What is the badge?',set()),
                ({'Cedar':{'badge':'violet triangle'},'Birch':{'badge':'silver circle'}},'Check the badge for Project Cedar.',{'Cedar'})]:
            world=RecordWorld(records);captured=[]
            def execute(payload):
                captured.append(payload)
                return dict(id='scoped',status='completed',model='local-test',output=[dict(type='function_call',
                    name='respond_to_user',call_id='reply',arguments=json.dumps(dict(route='clarify',content='Please identify an available project.')))])
            result=run_worker([dict(role='user',text=user)],world,'current',model='local-test',
                dispatch_protocol='required-action',response_executor=execute,scope_read_tools=True)
            definitions={d['name']:d for d in captured[0]['tools']}
            if expected:self.assertEqual(set(definitions['lookup_record']['parameters']['properties']['project']['enum']),expected)
            else:self.assertNotIn('lookup_record',definitions)
            self.assertEqual(result['receipts'],[]);self.assertEqual(world.receipts,[])

    def test_explicit_local_executor_needs_no_remote_credential_or_http(self):
        world=RecordWorld({});calls=[]
        def execute(payload):
            calls.append(payload)
            return {'id':'local-proof','status':'completed','model':'local-test','output':[
                {'type':'function_call','name':'respond_to_user','call_id':'reply',
                 'arguments':json.dumps({'route':'answer','content':'Three sides.'})}]}
        with patch.dict('os.environ',{'OPENAI_API_KEY':''}):
            result=run_worker([{'role':'user','text':'Triangle sides?'}],world,'current',model='local-test',
                dispatch_protocol='required-action',response_executor=execute)
        self.assertEqual(result['response_provider'],'custom-executor');self.assertEqual(calls[0]['model'],'local-test')
        self.assertEqual(result['events'][0]['model'],'local-test');self.assertEqual(result['receipts'],[])

    def test_state_aware_cancellation_can_fulfil_an_ordinary_reply_without_cancelling_it(self):
        world=RecordWorld({'Cedar':{'badge':'violet triangle'}})
        old=world.read('lookup','Cedar','badge');ordinary=world.reserve('ordinary');world.finish('ordinary',ordinary)
        revision=world.reserve('latest');pending=[{'id':'ordinary','request_text':'Opposite of clockwise?','text':'Counterclockwise.'}]
        outputs=[{'id':'call','status':'completed','output':[{'type':'function_call','name':'cancel_lookup','call_id':'cancel',
            'arguments':json.dumps({'task_ids':['lookup']})}]},
            {'id':'reply','status':'completed','output':[{'type':'message','content':[{'type':'output_text',
             'text':json.dumps({'route':'answer','content':'Cancelled. The opposite is counterclockwise.','consume_pending_speech_ids':['ordinary']})}]}]}]
        def post(*args,**kwargs):return types.SimpleNamespace(status_code=200,json=lambda:outputs.pop(0))
        with patch.dict('sys.modules',{'requests':types.SimpleNamespace(post=post)}),patch.dict('os.environ',{'OPENAI_API_KEY':'test-placeholder'}):
            result=run_worker([{'role':'user','text':'Cancel the lookup but finish your answer.'}],world,'latest',
                expected_revision=revision,dispatch_protocol='state-aware',pending_speech=pending)
        self.assertEqual(result['route'],'cancel');self.assertEqual(result['consume_pending_speech_ids'],['ordinary'])
        self.assertFalse(world.deliverable(old));self.assertFalse(world.tasks['ordinary']['cancelled'])
        self.assertEqual(result['receipts'][0]['task_ids'],['lookup'])

    def test_state_aware_reply_cannot_discard_an_unknown_ordinary_answer(self):
        world=RecordWorld({})
        def post(*args,**kwargs):
            return types.SimpleNamespace(status_code=200,json=lambda:{'id':'bad','status':'completed','output':[
                {'type':'function_call','name':'respond_to_user','call_id':'bad',
                 'arguments':json.dumps({'route':'answer','content':'Done.','consume_pending_speech_ids':['unknown']})}]})
        with patch.dict('sys.modules',{'requests':types.SimpleNamespace(post=post)}),patch.dict('os.environ',{'OPENAI_API_KEY':'test-placeholder'}):
            with self.assertRaisesRegex(WorkerContractError,'unknown, duplicate'):
                run_worker([{'role':'user','text':'Continue.'}],world,'latest',dispatch_protocol='state-aware')
        self.assertEqual(world.receipts,[])

    def test_required_action_still_rejects_available_project_without_user_identification(self):
        world=RecordWorld({'Elm North':{'locker label':'copper kite'}})
        revision=world.reserve('task')
        def post(*args,**kwargs):
            return types.SimpleNamespace(status_code=200,json=lambda:{'id':'bad','status':'completed','output':[
                {'type':'function_call','name':'lookup_record','call_id':'bad',
                 'arguments':json.dumps({'project':'Elm North','field':'locker label'})}]})
        with patch.dict('sys.modules',{'requests':types.SimpleNamespace(post=post)}),patch.dict('os.environ',{'OPENAI_API_KEY':'test-placeholder'}):
            with self.assertRaisesRegex(WorkerContractError,'not identified') as caught:
                run_worker([{'role':'user','text':"What's the locker label?"}],world,'task',expected_revision=revision,dispatch_protocol='required-action')
        self.assertEqual(world.receipts,[])
        self.assertEqual(caught.exception.result['arguments']['project'],'Elm North')

    def test_required_action_answers_in_one_call_without_fabricating_a_receipt(self):
        world=RecordWorld({});payloads=[]
        def post(*args,**kwargs):
            payloads.append(kwargs['json'])
            return types.SimpleNamespace(status_code=200,json=lambda:{'id':'reply','status':'completed','output':[
                {'type':'function_call','name':'respond_to_user','call_id':'reply1',
                 'arguments':json.dumps({'route':'answer','content':'Light travels faster than sound.'})}]})
        with patch.dict('sys.modules',{'requests':types.SimpleNamespace(post=post)}),patch.dict('os.environ',{'OPENAI_API_KEY':'test-placeholder'}):
            result=run_worker([{'role':'user','text':'Why lightning first?'}],world,'task',dispatch_protocol='required-action')
        self.assertEqual(len(payloads),1);self.assertEqual(result['route'],'answer')
        self.assertEqual(result['receipts'],[]);self.assertEqual(world.receipts,[])
        self.assertEqual(payloads[0]['tool_choice'],'required');self.assertNotIn('text',payloads[0])
        self.assertEqual({t['name'] for t in payloads[0]['tools']},{'respond_to_user','lookup_record','cancel_lookup'})

    def test_required_action_cannot_return_plan_or_claim_a_private_read_in_reply_action(self):
        for output in [[],[{'type':'function_call','name':'respond_to_user','call_id':'bad',
                           'arguments':json.dumps({'route':'delegate','content':'Done.'})}]]:
            world=RecordWorld({'Cedar':{'badge':'violet triangle'}})
            def post(*args,**kwargs):return types.SimpleNamespace(status_code=200,json=lambda:{'id':'bad','status':'completed','output':output})
            with patch.dict('sys.modules',{'requests':types.SimpleNamespace(post=post)}),patch.dict('os.environ',{'OPENAI_API_KEY':'test-placeholder'}):
                with self.assertRaises(WorkerContractError):
                    run_worker([{'role':'user','text':'Check Cedar.'}],world,'task',dispatch_protocol='required-action')
            self.assertEqual(world.receipts,[])

    def test_completed_request_is_not_pending_or_cancellable_and_old_revision_cannot_close_new(self):
        world=RecordWorld({'Cedar':{'badge':'violet triangle'}})
        receipt=world.read('lookup','Cedar','badge');rev=receipt['revision']
        world.finish('lookup',rev)
        self.assertFalse(world.deliverable(receipt))
        with self.assertRaises(ValueError):world.cancel('lookup')
        with self.assertRaises(WorkerSuperseded):world.read('lookup','Cedar','badge',expected_revision=rev)
        next_rev=world.reserve('lookup')
        with self.assertRaises(WorkerSuperseded):world.finish('lookup',rev)
        self.assertFalse(world.tasks['lookup']['closed'])
        self.assertGreater(next_rev,rev)

    def test_concurrent_cancel_rejection_does_not_touch_unrelated_work(self):
        world = RecordWorld({'Cedar':{'badge':'violet triangle'},'Birch':{'badge':'silver circle'}})
        old = world.read('old','Cedar','badge')
        other = world.read('other','Birch','badge')
        revision = world.reserve('dispatch')
        def post(*args,**kwargs):
            world.cancel('old')  # Another dispatch wins after the task snapshot.
            return types.SimpleNamespace(status_code=200,json=lambda:{
                'id':'race','status':'completed','output':[{'type':'function_call',
                    'name':'cancel_lookup','call_id':'cancel',
                    'arguments':json.dumps({'task_ids':['old']})}]})
        with patch.dict('sys.modules',{'requests':types.SimpleNamespace(post=post)}),patch.dict('os.environ',{'OPENAI_API_KEY':'test-placeholder'}):
            with self.assertRaisesRegex(WorkerContractError,'no longer active') as caught:
                run_worker([{'role':'user','text':'Cancel Cedar.'}],world,'dispatch',expected_revision=revision)
        self.assertEqual(caught.exception.result['receipts'],[])
        self.assertFalse(world.deliverable(old))
        self.assertTrue(world.deliverable(other))
        self.assertFalse(world.tasks['dispatch']['cancelled'])

    def test_unavailable_record_becomes_failed_reply_without_execution(self):
        from evals.context_backend_failure import notice
        for project,field in [('Cedar','missing'),('Birch','badge')]:
            with self.subTest(project=project,field=field):
                world = RecordWorld({'Cedar':{'badge':'violet triangle'}})
                revision = world.reserve('task')
                call = {'type':'function_call','name':'lookup_record','call_id':'bad',
                    'arguments':json.dumps({'project':project,'field':field})}
                requests = []
                def post(*args,**kwargs):
                    requests.append(kwargs)
                    return types.SimpleNamespace(status_code=200,json=lambda:{
                        'id':'bad','status':'completed','output':[call]})
                with patch.dict('sys.modules',{'requests':types.SimpleNamespace(post=post)}),patch.dict('os.environ',{'OPENAI_API_KEY':'test-placeholder'}):
                    with self.assertRaises(WorkerContractError) as caught:
                        run_worker([{'role':'user','text':'Check '+project+' '+field}],
                            world,'task',expected_revision=revision)
                result = notice(caught.exception,'task')
                self.assertEqual(result['route'],'error')
                self.assertEqual(result['receipts'],[])
                self.assertNotIn('violet triangle',result['content'])
                self.assertEqual(world.receipts,[])
                self.assertEqual(world.tasks['task']['phase'],'pending')
                self.assertEqual(len(requests),1)

    def test_cancel_selected_duplicates_preserves_unrelated_work(self):
        world = RecordWorld({"Cedar":{"badge":"violet triangle"},"Birch":{"badge":"silver circle"}})
        first = world.read("first","Cedar","badge")
        duplicate = world.read("duplicate","Cedar","badge")
        unrelated = world.read("unrelated","Birch","badge")
        receipt = world.cancel_many(["first","duplicate"])
        self.assertEqual(receipt["task_ids"],["first","duplicate"])
        self.assertFalse(world.deliverable(first))
        self.assertFalse(world.deliverable(duplicate))
        self.assertTrue(world.deliverable(unrelated))

    def test_cancel_batch_failure_changes_nothing(self):
        world = RecordWorld({"Cedar":{"badge":"violet triangle"}})
        first = world.read("first","Cedar","badge")
        with self.assertRaises(ValueError):
            world.cancel_many(["first","unknown"])
        self.assertTrue(world.deliverable(first))

    def test_cancel_dispatch_cannot_cancel_itself(self):
        world = RecordWorld({"Cedar":{"badge":"violet triangle"}})
        record = world.read("lookup","Cedar","badge")
        revision = world.reserve("cancel-dispatch")
        responses = [{"id":"execute","status":"completed","output":[{"type":"function_call",
            "name":"cancel_lookup","call_id":"cancel1","arguments":json.dumps({"task_id":"cancel-dispatch"})}]},
            {"id":"finish","status":"completed","output":[{"type":"message","content":[{
                "type":"output_text","text":json.dumps({"route":"cancel","content":"Cancelled."})}]}]}]
        def post(*args,**kwargs):
            return types.SimpleNamespace(status_code=200,json=lambda:responses.pop(0))
        with patch.dict("sys.modules",{"requests":types.SimpleNamespace(post=post)}),patch.dict("os.environ",{"OPENAI_API_KEY":"test-placeholder"}):
            with self.assertRaises(WorkerContractError):
                run_worker([{"role":"user","text":"Cancel that lookup."}],world,"cancel-dispatch",expected_revision=revision)
        self.assertTrue(world.deliverable(record))
        self.assertFalse(world.tasks["cancel-dispatch"]["cancelled"])

    def test_cancel_rejects_result_that_completed_before_delivery(self):
        world = RecordWorld({"Cedar":{"badge":"violet triangle"}})
        old = world.read("task","Cedar","badge")
        self.assertTrue(world.deliverable(old))
        world.cancel("task")
        self.assertFalse(world.deliverable(old))

    def test_correction_rejects_old_project_result(self):
        world = RecordWorld({"Cedar":{"badge":"violet triangle"},"Birch":{"badge":"silver circle"}})
        old = world.read("task","Cedar","badge")
        current = world.read("task","Birch","badge")
        self.assertFalse(world.deliverable(old))
        self.assertTrue(world.deliverable(current))
        self.assertEqual(current["value"],"silver circle")

    def test_missing_identification_cannot_execute(self):
        world = RecordWorld({"Cedar":{"badge":"violet triangle"}})
        with self.assertRaises(ValueError):
            world.read("task","unknown","badge")
        self.assertEqual(world.receipts,[])

    def test_cancel_before_record_execution_cannot_resurrect_task(self):
        world = RecordWorld({"Cedar":{"badge":"violet triangle"}})
        revision = world.reserve("task")
        world.cancel("task")
        with self.assertRaises(WorkerSuperseded):
            world.read("task","Cedar","badge",expected_revision=revision)
        self.assertEqual(world.receipts,[])

    def test_delegation_acknowledgement_requires_real_execution(self):
        world = RecordWorld({"Cedar":{"badge":"violet triangle"}})
        message = lambda route,content: {"type":"message","content":[{"type":"output_text",
            "text":json.dumps({"route":route,"content":content})}]}
        responses = [
            {"id":"plan","status":"completed","output":[message("delegate","I will check.")]},
            {"id":"execute","status":"completed","output":[{"type":"function_call","name":"lookup_record",
                "call_id":"call1","arguments":json.dumps({"project":"Cedar","field":"badge"})}]},
            {"id":"finish","status":"completed","output":[message("answer","The badge is violet triangle.")]},
        ]
        payloads = []
        def post(url,headers,json,timeout):
            payloads.append(json)
            return types.SimpleNamespace(status_code=200,json=lambda:responses.pop(0))
        with patch.dict("sys.modules",{"requests":types.SimpleNamespace(post=post)}),patch.dict("os.environ",{"OPENAI_API_KEY":"test-placeholder"}):
            result = run_worker([{"role":"user","text":"Check Cedar's badge"}],world,"task")
        self.assertEqual(len(world.receipts),1)
        self.assertEqual(result["receipts"][0]["value"],"violet triangle")
        self.assertEqual(result["route"],"delegate")
        self.assertEqual(result["final_response_route"],"answer")
        self.assertEqual(payloads[1]["tool_choice"],{"type":"function","name":"lookup_record"})
        self.assertNotIn("violet triangle",str(payloads[:2]))
        self.assertIn("violet triangle",str(payloads[2]))

    def test_available_project_or_assistant_guess_cannot_authorize_read(self):
        world = RecordWorld({"Cedar North":{"badge":"violet triangle"}})
        call = {"type":"function_call", "name":"lookup_record", "call_id":"call1",
            "arguments":json.dumps({"project":"Cedar North", "field":"badge"})}
        def post(*args, **kwargs):
            return types.SimpleNamespace(status_code=200,json=lambda:{"id":"wrong", "status":"completed", "output":[call]})
        context = [{"role":"user", "text":"What is the badge?"},
                   {"role":"assistant", "text":"It must be Cedar North."}]
        with patch.dict("sys.modules",{"requests":types.SimpleNamespace(post=post)}),patch.dict("os.environ",{"OPENAI_API_KEY":"test-placeholder"}):
            with self.assertRaisesRegex(WorkerContractError, "not identified"):
                run_worker(context,world,"task")
        self.assertEqual(world.receipts, [])

    def test_actual_tool_receipt_does_not_make_blank_backend_text_deliverable(self):
        world = RecordWorld({"Cedar":{"badge":"violet triangle"}})
        responses = [
            {"id":"execute","status":"completed","output":[{"type":"function_call","name":"lookup_record",
                "call_id":"call1","arguments":json.dumps({"project":"Cedar","field":"badge"})}]},
            {"id":"finish","status":"completed","output":[{"type":"message","content":[{
                "type":"output_text","text":json.dumps({"route":"answer","content":"  "})}]}]},
        ]
        def post(*args, **kwargs):
            return types.SimpleNamespace(status_code=200,json=lambda:responses.pop(0))
        with patch.dict("sys.modules",{"requests":types.SimpleNamespace(post=post)}),patch.dict("os.environ",{"OPENAI_API_KEY":"test-placeholder"}):
            with self.assertRaisesRegex(WorkerContractError, 'empty result text'):
                run_worker([{"role":"user","text":"Check Cedar's badge"}],world,"task")
        self.assertEqual(world.receipts[0]['value'], 'violet triangle')

    def test_user_deltas_are_reconstructed_without_invented_spaces(self):
        context = [{"speaker":"user", "text":part} for part in ("Project Ce", "dar", " North.")]
        self.assertTrue(project_identified_by_user(context, "Cedar North"))
        self.assertFalse(project_identified_by_user(context, "Cedar South"))


if __name__ == "__main__":
    unittest.main()
