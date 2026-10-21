"""Artificial WebSocket transport fixtures; no API/model/teacher approval."""
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals import native_control_teacher as teacher
from evals.native_control_teacher_artifacts import input_availability, verify_record


class Clock:
    def __init__(self):
        self.value = 100.

    def now(self):
        return self.value

    async def sleep(self, seconds):
        self.value += seconds
        await asyncio.sleep(0)


class Socket:
    def __init__(self, early_close=False, wrong_model=False, invalid_audio=False):
        self.queue = asyncio.Queue()
        self.sent = []
        self.early_close = early_close
        self.wrong_model = wrong_model
        self.invalid_audio = invalid_audio
        self.input_count = 0
        self.output = b'\x0b\x00\x16\x00'

    async def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        if message['type'] == 'session.start':
            self.queue.put_nowait(dict(type='session.started', session=dict(
                id='fixture-session', model='wrong-model' if self.wrong_model else teacher.MODEL,
                client_secret='fixture-secret-not-a-credential')))
        elif message['type'] == 'session.input_audio.append':
            self.input_count += 1
            if self.input_count == 1:
                self.queue.put_nowait(dict(type='session.output_transcript.delta', delta='Fixture words.'))
                self.queue.put_nowait(dict(type='session.output_audio.delta', delta=base64.b64encode(
                    b'x' if self.invalid_audio else self.output).decode()))
                if self.early_close:
                    self.queue.put_nowait(dict(type='session.closed', usage=dict(fixture_only=True)))
        elif message['type'] == 'session.close':
            self.queue.put_nowait(dict(type='session.closed', usage=dict(fixture_only=True)))
        await asyncio.sleep(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return json.dumps(await self.queue.get())


class Transport(unittest.IsolatedAsyncioTestCase):
    async def test_every_input_sample_and_output_byte_are_preserved_without_approval(self):
        import numpy as np
        clock, socket, events, spool = Clock(), Socket(), [], bytearray()
        samples = np.arange(1920 + 15 * 24000, dtype='int16')
        report, audio, packets, transcripts = await teacher.capture_stream(
            socket, samples, 1920, events.append, now=clock.now, sleep=clock.sleep,
            audio_sink=spool.extend)
        self.assertTrue(report['session_finalized'])
        self.assertEqual(report['errors'], [])
        self.assertEqual(report['sent_input_samples'], len(samples))
        self.assertFalse(report['training_approved'])
        actual = b''.join(base64.b64decode(m['audio']) for m in socket.sent if m['type'] == 'session.input_audio.append')
        self.assertEqual(actual, samples.tobytes())
        input_availability(events, len(samples), actual)
        self.assertEqual(audio, socket.output)
        self.assertEqual(audio, bytes(spool))
        self.assertEqual(len(packets), 1)
        self.assertEqual(transcripts[0]['text'], 'Fixture words.')
        self.assertEqual(events[0]['session']['client_secret'], '<REDACTED>')
        self.assertEqual(report['observed_until_s'], next(e['received_at_s'] for e in events if e['type'] == 'session.closed'))

    async def test_early_close_wrong_model_and_invalid_pcm_never_complete(self):
        import numpy as np
        samples = np.zeros(1920 + 15 * 24000, dtype='int16')
        for option in ('early_close', 'wrong_model', 'invalid_audio'):
            clock, socket, events = Clock(), Socket(**{option: True}), []
            report, _, _, _ = await teacher.capture_stream(socket, samples, 1920, events.append,
                                                          now=clock.now, sleep=clock.sleep)
            complete = report['session_finalized'] and not report['errors'] and report['sent_input_samples'] == len(samples)
            self.assertFalse(complete, option)

    async def test_record_reconstruction_and_connection_failure_keep_raw_evidence(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'source'
            sid = 'native-control3-train-000'
            (raw / sid).mkdir(parents=True)
            sf.write(raw / sid / 'stem_user.wav', np.full(1920, .1, dtype='float32'), 24000, subtype='FLOAT')
            (raw / 'raw-source.json').write_text('{"fixture_only":true}')
            row = dict(id=sid, frames=1920)
            original = teacher.capture_stream
            for fail_exit in (False, True):
                clock, socket = Clock(), Socket()
                class Connection:
                    async def __aenter__(self):
                        return socket

                    async def __aexit__(self, *args):
                        if fail_exit:
                            raise RuntimeError('fixture-key-never-print-this-message')

                def connector(url, **kwargs):
                    self.assertEqual(url, 'wss://api.openai.com/v1/live/sessions')
                    self.assertEqual(kwargs['additional_headers']['Authorization'], 'Bearer fixture-key')
                    return Connection()

                async def runner(*args, **kwargs):
                    return await original(*args, **kwargs, now=clock.now, sleep=clock.sleep)

                folder = root / ('failed' if fail_exit else 'complete')
                with patch.object(teacher, 'capture_stream', side_effect=runner):
                    record = await teacher.capture_one(raw, row, folder, 'source-audit-fixture',
                                                       'fixture-key', connector=connector)
                self.assertEqual((folder / 'output-stream.pcm').read_bytes(), socket.output)
                self.assertEqual(set(record['artifacts']), set(teacher.ARTIFACTS))
                self.assertNotIn('fixture-key', (folder / 'environment.json').read_text())
                self.assertNotIn('never-print-this-message', (folder / 'report.json').read_text())
                if fail_exit:
                    self.assertFalse(record['transport_complete'])
                else:
                    result = verify_record(raw, row, folder, 'source-audit-fixture', record)
                    self.assertTrue(result['transport_integrity_passed'])
                    self.assertFalse(result['behavior_or_label_quality_approved'])
                    pcm, rate = sf.read(folder / 'model_out.wav', dtype='int16')
                    pcm[-1] = 0
                    sf.write(folder / 'model_out.wav', pcm, rate)
                    record['artifacts']['model_out.wav'] = hashlib.sha256((folder / 'model_out.wav').read_bytes()).hexdigest()
                    with self.assertRaisesRegex(ValueError, 'exact unbuffered original packet playout'):
                        verify_record(raw, row, folder, 'source-audit-fixture', record)


class LocalSetup(unittest.TestCase):
    def test_live_lease_prevents_overlap_and_stale_file_does_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'capture.lock'
            with teacher.collection_lock(path):
                with self.assertRaisesRegex(RuntimeError, 'still executing'):
                    with teacher.collection_lock(path):
                        self.fail('Duplicate collector acquired a live lease')
            with teacher.collection_lock(path):
                self.assertTrue(path.exists())

    def test_private_key_file_is_parsed_without_shell_execution_or_logging(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            path = Path(temporary) / 'backend.env'
            path.write_text('OPENAI_API_KEY=\n')
            path.chmod(0o600)
            with self.assertRaisesRegex(RuntimeError, 'not configured'):
                teacher.local_key(path)
            path.write_text('export OPENAI_API_KEY="fixture-value" # local only\n')
            self.assertEqual(teacher.local_key(path), 'fixture-value')
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, 'mode0600'):
                teacher.local_key(path)

    def test_error_details_and_authentication_fields_are_not_archived(self):
        result = teacher.safe_event(dict(type='error', error=dict(type='auth', code='invalid', message='fixture-secret')))
        self.assertNotIn('fixture-secret', json.dumps(result))
        self.assertFalse('model' in result)
        result = teacher.safe_event(dict(type='session.started', session=dict(api_key='fixture-secret')))
        self.assertEqual(result['session']['api_key'], '<REDACTED>')


if __name__ == '__main__':
    unittest.main()
