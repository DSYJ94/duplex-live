import unittest
from evals.recovery_worker import RecordWorld


class RecordGuards(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
