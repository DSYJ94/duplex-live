"""Explicit prompt/version and byte-preserving transport fixtures, not label approval."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals import native_control_teacher_exact as exact
from evals import native_control_teacher as original
from evals.harness.test_native_control_teacher import Clock, Socket


class ExactPromptTransport(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_concurrency_fails_before_reading_data_or_opening_network(self):
        for concurrency in (0, 3, True):
            with self.assertRaisesRegex(ValueError, 'bounded sessions'):
                await exact.collect(*(Path('absent-fixture') for _ in range(6)), 'fixture-key', Path('lease'),
                                    concurrency=concurrency)

    async def test_actual_start_uses_declared_prompt_and_original_recorder_preserves_pcm(self):
        import numpy as np
        import soundfile as sf
        class EchoSocket(Socket):
            async def send(self, raw):
                await super().send(raw)
                message = json.loads(raw)
                if message['type'] == 'session.start':
                    event = self.queue.get_nowait()
                    event['session']['instructions'] = message['session']['instructions']
                    self.queue.put_nowait(event)
        with tempfile.TemporaryDirectory() as temporary:
            root, clock, socket = Path(temporary), Clock(), EchoSocket()
            sid = exact.FAILED_IDS[0]
            (root / sid).mkdir()
            sf.write(root / sid / 'stem_user.wav', np.full(1920, .125, dtype='float32'), 24000, subtype='FLOAT')
            (root / 'raw-source.json').write_text('{"fixture_only":true}')
            class Connection:
                async def __aenter__(self): return socket
                async def __aexit__(self, *args): return None
            def connector(url, **kwargs): return Connection()
            capture = exact.capture_stream
            async def run(*args, **kwargs):
                return await capture(*args, **kwargs, now=clock.now, sleep=clock.sleep)
            folder = root / 'record'
            with patch.object(exact, 'capture_stream', side_effect=run):
                record = await exact.capture_one(root, dict(id=sid, frames=1920), folder, 'fixture-audit',
                                                 'fixture-key-not-a-credential', connector=connector)
            self.assertTrue(record['transport_complete'])
            start = next(e for e in socket.sent if e['type'] == 'session.start')
            self.assertEqual(start['session'], exact.configuration())
            self.assertNotEqual(start['session']['instructions'], original.configuration()['instructions'])
            self.assertEqual((folder / 'output-stream.pcm').read_bytes(), socket.output)
            self.assertNotIn('fixture-key-not-a-credential', (folder / 'environment.json').read_text())
            exact.verify_exact_record(root, dict(id=sid, frames=1920), folder, 'fixture-audit', record)
            environment = json.loads((folder / 'environment.json').read_text())
            environment['session_config'] = original.configuration()
            (folder / 'environment.json').write_text(json.dumps(environment))
            record['artifacts']['environment.json'] = exact.sha256(folder / 'environment.json')
            with self.assertRaisesRegex(ValueError, 'instructions or source bindings'):
                exact.verify_exact_record(root, dict(id=sid, frames=1920), folder, 'fixture-audit', record)

    def test_prompt_keeps_native_teacher_model_voice_and_authored_words(self):
        old, new = original.configuration(), exact.configuration()
        self.assertEqual({k:v for k,v in old.items() if k != 'instructions'},
                         {k:v for k,v in new.items() if k != 'instructions'})
        self.assertIn('Reflected light carries light to your eyes.', new['instructions'])
        self.assertIn('all seven words', new['instructions'])
        self.assertIn('Point at the window.', new['instructions'])
        self.assertIn('Touch the table.', new['instructions'])
        self.assertEqual(len(exact.FAILED_IDS), 2)

    def test_new_quality_schema_cannot_bypass_complete40_gate(self):
        from evals import native_control_semantic as semantic
        from evals import native_control_teacher_exact_quality as quality
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'quality.json'
            path.write_text(json.dumps(dict(schema=quality.SCHEMA)))
            with patch.object(quality, 'verify', side_effect=ValueError('Whole40 failed')) as verifier:
                with self.assertRaisesRegex(ValueError, 'Whole40 failed'):
                    semantic.verify_quality(root, root, root, path)
                verifier.assert_called_once_with(root, root, root, path)
            path.write_text(json.dumps(dict(schema='unapproved-fixture')))
            with self.assertRaisesRegex(ValueError, 'approved control40'):
                semantic.verify_quality(root, root, root, path)


if __name__ == '__main__':
    unittest.main()
