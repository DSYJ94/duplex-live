import unittest
from evals.gpt_live_replay import audio_sample_count


class ApprovedAudioLength(unittest.TestCase):
    def test_actual_verified_waveform_round_trip(self):
        self.assertEqual(audio_sample_count(478080,24000,478080/24000),478080)

    def test_missing_samples_still_reject(self):
        with self.assertRaises(ValueError):
            audio_sample_count(478079,24000,19.92)


if __name__ == "__main__":
    unittest.main()
