import unittest
import json
import types
from unittest.mock import patch
from evals.recovery_worker import RecordWorld,run_worker,WorkerSuperseded,WorkerContractError,project_identified_by_user


class RecordGuards(unittest.TestCase):
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
