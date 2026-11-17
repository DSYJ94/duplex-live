import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from training.personaplex.train_failure_audio import extract, prepare
from training.personaplex.train_failure_inputs import build


class TrainFailureAudioTest(unittest.TestCase):
    def fixture(self, root):
        wave = root / 'scene.wav'
        # Teacher is unmistakably distinct and lasts longer than the user.
        pcm = np.column_stack((np.full(24, .75), np.arange(24) / 100)).astype('float32')
        sf.write(wave, pcm, 24000, subtype='FLOAT')
        side = wave.with_suffix('.json')
        side.write_text(json.dumps(dict(fold='TRAIN')))
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        return dict(id='train-scene', fold='TRAIN', question='No answer supplied',
                    source_wave='scene.wav', source_wave_sha256=sha(wave),
                    source_sidecar_sha256=sha(side), user_channel=1, user_frames=12,
                    response_tail_seconds=60, required_user_only_extraction=True), pcm

    def test_exact_whole_user_and_fixed_tail_never_teacher_or_teacher_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record, original = self.fixture(root)
            result = extract(root, record, root / 'user.wav')
            saved, rate = sf.read(root / 'user.wav', dtype='float32')
            np.testing.assert_array_equal(saved[:12], original[:12, 1])
            self.assertEqual(len(saved), 12 + 60 * 24000)
            self.assertFalse(saved[12:].any())
            self.assertEqual(rate, 24000)
            self.assertFalse(result['teacher_audio_in_model_input'])
            self.assertFalse(result['native_inference_started'])
            with self.assertRaises(FileExistsError):
                extract(root, record, root / 'user.wav')

    def test_changed_source_or_fold_cannot_enter_inference(self):
        for kind in ('wave', 'sidecar', 'fold', 'channel', 'frames', 'tail', 'gold', 'path'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                record, _ = self.fixture(root)
                if kind == 'wave': (root / 'scene.wav').write_bytes(b'changed')
                if kind == 'sidecar': (root / 'scene.json').write_text('{}')
                if kind == 'fold': record['fold'] = 'DEV'
                if kind == 'channel': record['user_channel'] = 0
                if kind == 'frames': record['user_frames'] = 25
                if kind == 'tail': record['response_tail_seconds'] = 8
                if kind == 'gold': record['answer'] = 'Never provide this'
                if kind == 'path': record['source_wave'] = '../scene.wav'
                with self.assertRaises(ValueError):
                    extract(root, record, root / 'user.wav')
                self.assertFalse((root / 'user.wav').exists())

    def test_complete_batch_rejects_changed_membership_before_any_extraction(self):
        from evals.harness.test_train_failure_inputs import TrainFailureInputsTest
        for change in ('subset', 'duplicate', 'dev', 'hash'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / 'source'
                (source / 'qualification').mkdir(parents=True)
                manifest, semantic = TrainFailureInputsTest().fixture()
                (source / 'manifest.json').write_text(json.dumps(manifest))
                (source / 'qualification/plan.json').write_text(json.dumps(semantic))
                sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
                value = build(manifest, semantic)
                value.update(source_root=str(source),
                             source_manifest_sha256=sha(source / 'manifest.json'),
                             semantic_parent_sha256=sha(source / 'qualification/plan.json'))
                if change == 'subset': value['records'].pop()
                if change == 'duplicate': value['records'][1] = value['records'][0]
                if change == 'dev': value['records'][0]['fold'] = 'DEV'
                inputs = root / 'input.json'
                inputs.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    prepare(inputs, '0' * 64 if change == 'hash' else sha(inputs), root / 'out')
                self.assertFalse((root / 'out').exists())
