"""PCM ownership and separated control-history fixtures; not model approval."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals import native_control_convert as convert


class NativeControlConversion(unittest.TestCase):
    def test_pcm_ownership_and_internal_silence_are_preserved(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            user = np.full(24000, .125, dtype='float32')
            assistant = np.zeros(24000 + 3840, dtype='float32')
            assistant[:1920] = -.25
            assistant[-1920:] = .25
            sf.write(root / 'user.wav', user, 24000, subtype='FLOAT')
            sf.write(root / 'assistant.wav', assistant, 24000, subtype='FLOAT')
            u, a, clock = convert.padded_stems(root / 'user.wav', root / 'assistant.wav', 1.5)
            self.assertTrue(np.array_equal(u[:len(user)], user))
            self.assertTrue(np.array_equal(a[:len(assistant)], assistant))
            self.assertEqual(np.count_nonzero(a[1920:24000]), 0)
            self.assertEqual(clock['channels'], ['assistant', 'user'])
            self.assertEqual(len(a) % 1920, 0)
            self.assertGreaterEqual(len(a) / 24000, 1.5)
            self.assertLess(len(a) / 24000 - 1.5, .08)
            self.assertFalse(clock['gain_changed'])
            self.assertFalse(clock['internal_silence_removed'])
            with self.assertRaises(ValueError):
                convert.padded_stems(root / 'user.wav', root / 'assistant.wav', 1.)

    def test_pause_and_resume_remain_separate_measured_answer_phases(self):
        from data.synthetic.native_control_repair import build
        from evals.harness.test_native_control_quality import ControlQuality
        from evals.native_control_teacher_checks import assess
        scene = next(c for c in build()['cases'] if c['family'] == 'pause-resume')
        source, speech, pcm, rate, log, session, checks = ControlQuality().fixture(scene)
        decision = assess(scene, source, speech, pcm, rate, log, session, checks)
        self.assertTrue(decision['passed'])
        for e in source['events']:
            e['actual_start_ms'] = e['actual_start_sample'] / 24
            e['actual_end_ms'] = e['actual_end_sample'] / 24
        timeline, side, maximum = convert.measured_timeline(scene, source, dict(clips=checks),
            dict(speech=speech, checks=decision, actual_input_packet_availability=log[:-2]))
        assistant = [e for e in timeline.events if e.ch == 'assistant']
        self.assertEqual(len(assistant), 2)
        self.assertEqual(assistant[0].text, scene['turns'][1]['text'])
        self.assertEqual(assistant[1].text, scene['turns'][4]['text'])
        self.assertGreater(assistant[1].actual_start_ms, decision['quiet_end_s'] * 1000)
        self.assertLessEqual(maximum, 200)
        self.assertEqual(len(side['alignments']), sum(len(e.text.split()) for e in assistant))
        self.assertEqual(side['reference_arrivals'], [])

    def test_failed_whole_semantic_approval_creates_no_converted_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'teacher.json').write_text('{}')
            with patch.object(convert, 'verify_semantic', side_effect=ValueError('Complete40 rejected')):
                with self.assertRaisesRegex(ValueError, 'Complete40 rejected'):
                    convert.convert(root, root, root, root, root, root / 'out')
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
