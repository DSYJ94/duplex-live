import unittest
from evals.personaplex_policy import action_targets


class NativePolicyTests(unittest.TestCase):
    def test_legacy_approved_cancel_teacher_retains_its_actual_action_kind(self):
        timeline = {"meta":{"kind":"cancel", "teacher_audio":True,
            "reference_arrivals":[{"ready_at_s":1.2, "text":"Cancelled.", "task_id":"cancel"}]},
            "events":[{"ch":"assistant", "kind":"marker", "start_ms":80,
                "note":"delegate actual teacher worker"},
                {"ch":"assistant", "kind":"marker", "start_ms":800,
                "note":"delegate actual teacher worker"}]}
        codes = [3]*20
        codes[1] = codes[10] = 4
        self.assertEqual(action_targets(codes,timeline)[10], 2)
        self.assertEqual(action_targets(codes,timeline)[1], 1)

    def test_lookup_and_cancel_keep_exact_distinct_clocks(self):
        codes = [3, 4, 10, 3, 4, 20]
        timeline = {"events": [
            {"ch": "assistant", "kind": "marker", "start_ms": 320, "note": "delegate cancellation"},
            {"ch": "assistant", "kind": "marker", "start_ms": 80, "note": "delegate lookup"}]}
        self.assertEqual(action_targets(codes, timeline), [0, 1, 0, 0, 2, 0])

    def test_misaligned_or_missing_markers_fail_closed(self):
        for time in (0, 79, 160):
            with self.assertRaises(ValueError):
                action_targets([3, 4], {"events": [
                    {"ch": "assistant", "kind": "marker", "start_ms": time, "note": "delegate"}]})
        with self.assertRaises(ValueError):
            action_targets([3, 4], {"events": []})


if __name__ == "__main__":
    unittest.main()
