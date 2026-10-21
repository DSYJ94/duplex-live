"""Actual PCM clock guards, with artificial word records and no API calls."""
import json
from pathlib import Path
import tempfile
import unittest

from evals.native_control_teacher_quality import speech_record


class TeacherClock(unittest.TestCase):
    def test_asr_cannot_place_first_word_in_unrecorded_leading_voice(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pcm = np.zeros(3 * 24000, dtype='float32')
            pcm[2 * 24000:] = .1
            sf.write(root / 'model_out.wav', pcm, 24000, subtype='FLOAT')
            (root / 'transcripts.json').write_text(json.dumps([
                dict(speaker='assistant', text='First '), dict(speaker='user', text='unscored'),
                dict(speaker='assistant', text='step.')]))
            for start, expected in ((.1, False), (1.79, False), (1.81, True), (2., True)):
                response = dict(text='First step.', segments=[dict(words=[
                    dict(text='First', start=start, end=2.2),
                    dict(text='step.', start=2.3, end=2.5)])])
                speech, actual, rate = speech_record(root, response)
                self.assertEqual(speech['model_text'], 'First step.')
                self.assertEqual(speech['word_starts_within_recorded_audio'], expected)
                self.assertTrue(np.array_equal(actual, pcm))
                self.assertEqual(rate, 24000)
                self.assertFalse(speech['expected_text_prompt_used'])


if __name__ == '__main__':
    unittest.main()
