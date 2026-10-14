import unittest
from evals.recovery_result_scope import private_result_frames


class PrivateResultScope(unittest.TestCase):
    def fixture(self):
        return {"timeline": {"meta": {"phrase": "yellow circle"}, "events": [
            {"ch": "assistant", "kind": "utterance", "actual_start_ms": 0,
             "actual_end_ms": 720, "text": "Checking yellow circle light"}]},
            "alignments": [["Checking", [0, .07], "SPEAKER_MAIN"],
                           ["yellow", [.16, .23], "SPEAKER_MAIN"],
                           ["circle", [.4, .47], "SPEAKER_MAIN"],
                           ["light", [.64, .71], "SPEAKER_MAIN"]],
            "reference_arrivals": [{"ready_at_s": .12, "text": "yellow circle"}]}

    def test_merged_teacher_weights_only_the_checked_phrase(self):
        vocab = {"Checking": [5], "yellow": [6, 7], "circle": [8, 9], "light": [10]}
        actual = private_result_frames(self.fixture(), [5, 3, 6, 7, 4, 8, 9, 3, 10], vocab.__getitem__)
        self.assertEqual(actual, [2, 3, 5, 6])

    def test_premature_phrase_fails_closed(self):
        sidecar = self.fixture()
        sidecar["reference_arrivals"][0]["ready_at_s"] = .3
        vocab = {"Checking": [5], "yellow": [6, 7], "circle": [8, 9], "light": [10]}
        with self.assertRaisesRegex(RuntimeError, "before.*ready"):
            private_result_frames(sidecar, [5, 3, 6, 7, 4, 8, 9, 3, 10], vocab.__getitem__)

    def test_dropped_native_subword_fails_closed(self):
        vocab = {"Checking": [5], "yellow": [6, 7], "circle": [8, 9], "light": [10]}
        with self.assertRaisesRegex(RuntimeError, "dropped or reordered"):
            private_result_frames(self.fixture(), [5, 3, 6, 3, 4, 8, 9, 3, 10], vocab.__getitem__)


if __name__ == "__main__":
    unittest.main()
