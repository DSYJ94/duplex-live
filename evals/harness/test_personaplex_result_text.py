import unittest
from evals.personaplex_result_text import WorkerTextQueue


class WorkerTextTests(unittest.TestCase):
    def test_only_completed_words_enter_native_slots(self):
        seen = []
        def encode(word):
            seen.append(word)
            return [5+len(seen)]
        queue = WorkerTextQueue(encode)
        receipt = {"task":"read"}
        queue.append("read", "yellow circle", receipt)
        result = []
        while (row := queue.pop(lambda r: True)) is not None:
            result.append(row)
        self.assertEqual(seen, ["yellow", "circle"])
        self.assertEqual([t for _,t in result[:8]], [6,3,3,0,7,3,3,0])
        self.assertTrue(all(task == "read" for task,_ in result))

    def test_cancelled_result_tail_cannot_enter_voice(self):
        queue = WorkerTextQueue(lambda word:[6])
        queue.append("lookup", "private value", {"task":"lookup"})
        self.assertEqual(queue.pop(lambda r:True), ("lookup",6))
        queue.append("cancel", "Cancelled.", None)
        self.assertEqual(queue.pop(lambda r:False), ("cancel",6))
        self.assertTrue(all(task == "cancel" for task,_,_ in queue.queue))

    def test_invalid_word_does_not_partially_enqueue(self):
        queue = WorkerTextQueue(lambda word:[6] if word == "good" else [4])
        with self.assertRaises(ValueError):
            queue.append("x", "good bad", None)
        self.assertEqual(len(queue.queue), 0)


if __name__ == "__main__":
    unittest.main()
