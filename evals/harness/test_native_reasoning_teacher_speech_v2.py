"""Actual explicit-region timestamp mapping and unchanged full-waveform gates."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evals import native_reasoning_teacher_speech_v2 as audit
from evals import native_reasoning_source_quality as quality


class RegionClockTests(unittest.TestCase):
    def test_all_active_blocks_remain_and_original_samples_are_untouched(self):
        pcm = np.zeros(24 * 24000, dtype=np.float32)
        pcm[8 * 24000:9 * 24000] = .1
        pcm[20 * 24000:21 * 24000] = .1
        original = pcm.copy(); spans = audit.regions(pcm, 24000)
        self.assertEqual(spans, [[7.84, 9.16], [19.84, 21.16]])
        np.testing.assert_array_equal(pcm, original)
        # Exercise installed Whisper's actual explicit-region clock converter;
        # this performs no neural inference and needs no model/GPU/download.
        module = importlib.import_module('whisper_timestamped.transcribe')
        import torch
        joined, actual, restore = module.remove_non_speech(
            torch.zeros(24 * 16000), method=[tuple(r) for r in spans])
        self.assertEqual(actual, [tuple(r) for r in spans])
        self.assertEqual(len(joined), int(2.64 * 16000))
        self.assertEqual(restore(.16, .66), [8., 8.5])
        self.assertEqual(restore(1.48, 1.98), [20., 20.5])

    def test_original_clipping_and_early_clocks_still_fail(self):
        pcm = np.zeros(10 * 24000, dtype=np.float32); pcm[8 * 24000:9 * 24000] = .1
        raw = dict(text='Eight tokens.', segments=[dict(words=[
            dict(text='Eight', start=.98, end=8.04), dict(text='tokens.', start=8.04, end=8.5)])])
        before = quality.assess('Eight tokens.', raw, pcm, 24000)
        self.assertFalse(before['passed']); self.assertIn('Word onset predates detectable original voice beyond200ms', before['reasons'])
        raw['segments'][0]['words'][0]['start'] = 8.
        self.assertTrue(quality.assess('Eight tokens.', raw, pcm, 24000)['passed'])
        pcm[0] = 1.; self.assertFalse(quality.assess('Eight tokens.', raw, pcm, 24000)['passed'])

    def test_missing_or_extra_words_still_fail(self):
        pcm = np.full(24000, .1, dtype=np.float32)
        raw = dict(text='Thank you.', segments=[dict(words=[
            dict(text='Thank', start=.1, end=.3), dict(text='you.', start=.3, end=.6)])])
        self.assertFalse(quality.assess('Eight tokens.', raw, pcm, 24000)['passed'])

    def test_whole32_plan_cannot_include_answers_or_partial_collection(self):
        plan = dict(schema=audit.PLAN_SCHEMA, policy=audit.POLICY,
                    code_sha256=audit.sha(Path(audit.__file__)), recordings=[
                        dict(id=f'native-reasoning4-train-{i:03d}',
                             wav=f'native-reasoning4-train-{i:03d}/model_out.wav',
                             wav_sha256='fixture', samples=24000, rate=24000) for i in range(32)])
        self.assertEqual(audit.validate_plan(plan), plan)
        partial = copy.deepcopy(plan); partial['recordings'].pop()
        with self.assertRaises(ValueError): audit.validate_plan(partial)
        reference = copy.deepcopy(plan); reference['recordings'][0]['expected'] = 'Eight tokens.'
        with self.assertRaises(ValueError): audit.validate_plan(reference)

    def test_all32_recomputed_rows_and_immutable_failure_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); failed = root / 'failed.json'; failed.write_text('{}')
            (root / 'recognition.json').write_text('{}')
            rows = [dict(id=f'native-reasoning4-train-{i:03d}', passed=True) for i in range(32)]
            args = (root, root, root, root, root, failed, root, root)
            with patch.object(audit, 'assess_all', return_value=rows):
                v = audit.finalize(*args)
                audit.verify(*args)
                v['results'] = v['results'][:-1]
                (root / 'teacher-speech-quality.json').write_text(json.dumps(v))
                with self.assertRaises(ValueError): audit.verify(*args)
                v['results'] = rows; (root / 'teacher-speech-quality.json').write_text(json.dumps(v))
                failed.write_text('{"rewritten":true}')
                with self.assertRaises(ValueError): audit.verify(*args)


if __name__ == '__main__': unittest.main()
