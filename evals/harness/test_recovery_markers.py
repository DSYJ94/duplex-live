import unittest
from evals.recovery_encode import marker_frame


class ControlTime(unittest.TestCase):
    def test_marker_uses_declared_time_without_an_audio_timestamp(self):
        self.assertEqual(marker_frame({"kind":"marker","text":"","start_ms":4800,"actual_start_ms":None},12.5),60)

    def test_speech_cannot_use_marker_contract(self):
        with self.assertRaises(ValueError):
            marker_frame({"kind":"utterance","text":"hello","start_ms":4800,"actual_start_ms":None},12.5)


if __name__ == "__main__":
    unittest.main()
