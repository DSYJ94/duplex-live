import unittest
import json
import types
from unittest.mock import patch
from evals.recovery_worker import RecordWorld,run_worker,WorkerSuperseded,WorkerContractError


class RecordGuards(unittest.TestCase):
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
            result = run_worker([{"text":"Check Cedar's badge"}],world,"task")
        self.assertEqual(len(world.receipts),1)
        self.assertEqual(result["receipts"][0]["value"],"violet triangle")
        self.assertEqual(result["route"],"delegate")
        self.assertEqual(result["final_response_route"],"answer")
        self.assertEqual(payloads[1]["tool_choice"],{"type":"function","name":"lookup_record"})
        self.assertNotIn("violet triangle",str(payloads[:2]))
        self.assertIn("violet triangle",str(payloads[2]))


if __name__ == "__main__":
    unittest.main()
