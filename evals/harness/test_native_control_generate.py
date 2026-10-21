"""Prevent a rejected credential from fanning out paid source requests."""
import json
import hashlib
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.synthetic.native_control_generate import collect, source_tasks
from data.synthetic.native_control_render import load_clip, sha256
from evals.native_control_pipeline import generation_complete
from data.synthetic.native_control_repair import build


class SourceCollectionFailure(unittest.TestCase):
    def test_rejected_probe_stops_collection_and_preserves_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / 'plan.json'
            plan.write_text(json.dumps(build()))
            calls = []

            def rejected(text, voice, path):
                calls.append((text, voice))
                raise RuntimeError('Speech API rejected request: HTTP 401')

            with self.assertRaisesRegex(RuntimeError, 'Source collection failed'):
                collect(plan, root / 'source', 'fixture-not-a-key', root / 'lease', generate=rejected)
            self.assertEqual(len(calls), 1)
            ledger = json.loads((root / 'source/generation.json').read_text())
            self.assertEqual(ledger['failures'][0]['http_status'], 401)
            self.assertEqual(ledger['status'], 'failed_preserved_unapproved')
            self.assertFalse(ledger['training_approved'])
            self.assertNotIn('fixture-not-a-key', (root / 'source/generation.json').read_text())
            with self.assertRaisesRegex(ValueError, 'NEW version'):
                collect(plan, root / 'source', 'fixture-not-a-key', root / 'lease', generate=rejected)
            self.assertEqual(len(calls), 1)

    def test_repair_retains139_originals_and_rejects_changed_copies(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / 'plan.json'
            plan.write_text(json.dumps(build()))
            parent = root / 'parent'
            parent.mkdir()
            tasks = source_tasks(build())
            rows = []

            def write(task, path, clipped=False):
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.full(4800, 1. if clipped else .1), 24000, subtype='PCM_16')
                path.with_suffix('.receipt.json').write_text(json.dumps(dict(
                    input_text=task['text'], voice=task['voice'], sha256=sha256(path),
                    source_model='fixture-speech-model', request_id='fixture-only')))

            missing = ('native-control3-train-031', 0)
            failed = ('native-control3-train-034', 2)
            for task in tasks[:141]:
                identity = task['case'], task['index']
                if identity == missing:
                    continue
                name = f'clips/{task["case"]}/{task["index"]:02d}_{task["role"]}.wav'
                write(task, parent / name, clipped=identity == failed)
                if identity != failed:
                    _, receipt = load_clip(parent / name, task)
                    rows.append(dict(task, path=name, **receipt))
            self.assertEqual(len(rows), 139)
            failed_path = parent / 'clips/native-control3-train-034/02_user.wav'
            pcm, _ = sf.read(failed_path)
            (parent / 'failed-waveform-diagnostic.json').write_text(json.dumps(dict(
                wav_sha256=sha256(failed_path), peak=float(np.abs(pcm).max()),
                samples_above_limit=int(np.sum(np.abs(pcm) > .995)))))
            (parent / 'generation.json').write_text(json.dumps(dict(
                status='failed_preserved_unapproved', training_approved=False,
                source_commit='fixture-parent', implementation_sha256=hashlib.sha256(b'fixture-source').hexdigest(),
                speech_helper_sha256=sha256(Path('data/synthetic/recovery_curriculum.py')),
                plan_sha256=sha256(plan), clips=rows)))
            calls = []

            def generate(text, voice, path, instructions=None):
                calls.append((text, instructions))
                write(dict(text=text, voice=voice), path)

            def git(command, **kwargs):
                return b'fixture-source' if command[1] == 'show' else 'fixture-current'

            with patch('data.synthetic.native_control_generate.subprocess.check_output', side_effect=git), contextlib.redirect_stdout(io.StringIO()):
                result = collect(plan, root / 'repair', 'fixture-not-a-key', root / 'lease',
                                 generate=generate, repair_source=parent)
                self.assertEqual(len(calls), 29)
                self.assertEqual(calls[0][0], 'Lantern.')
                self.assertIn('once', calls[0][1])
                self.assertEqual(len(result['clips']), 168)
                self.assertFalse(result['gain_changed'])
                self.assertFalse(result['training_approved'])
                self.assertTrue(generation_complete(root / 'repair', plan))
                for row in rows:
                    self.assertEqual(sha256(root / 'repair' / row['path']), row['waveform_sha256'])
                (root / 'repair' / rows[0]['path']).write_bytes(b'changed original')
                with self.assertRaisesRegex(ValueError, 'retained original'):
                    generation_complete(root / 'repair', plan)


if __name__ == '__main__':
    unittest.main()
