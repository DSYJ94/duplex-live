"""Counterfactual facts may change; the approved question and lineage may not."""
import json
import tempfile
import unittest
from pathlib import Path
from evals.teacher_world import apply_world, teacher_metadata


class TeacherWorldTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"world.json"
        self.meta = {"project":"Birch North", "field":"badge phrase", "phrase":"blue circle"}
        self.world = {"schema":"dl-teacher-world-v1", "source_scenario":"train-0",
            "source_ingestion_sha256":"sealed", "project":"Birch North", "field":"badge phrase", "phrase":"red star"}

    def save(self):
        self.path.write_text(json.dumps(self.world))

    def test_only_declared_value_changes(self):
        self.save()
        actual, receipt = apply_world(self.path, self.meta, "train-0", "sealed")
        self.assertEqual(actual["phrase"], "red star")
        self.assertEqual(self.meta["phrase"], "blue circle")
        record = {"source_row":{"scenario_id":"train-0"}, "world_override":receipt}
        self.assertEqual(teacher_metadata(record, self.meta, "sealed"), actual)
        self.world["phrase"] = "gold oval"
        self.save()
        with self.assertRaisesRegex(RuntimeError, "changed"):
            teacher_metadata(record, self.meta, "sealed")

    def test_question_lineage_and_distinct_value_required(self):
        for key,value in (("project","Other Project"), ("field","another field"),
                          ("source_scenario","test-0"), ("source_ingestion_sha256","different"),
                          ("phrase","blue circle")):
            with self.subTest(key=key):
                original = self.world[key]
                self.world[key] = value
                self.save()
                with self.assertRaises(RuntimeError):
                    apply_world(self.path, self.meta, "train-0", "sealed")
                self.world[key] = original


if __name__ == "__main__":
    unittest.main()
