import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from training.personaplex.train_failure_rollout import collect_case, shard


class Engine:
    frame_size = 1920

    def __init__(self): self.calls = 0

    def step(self, frame):
        self.calls += 1
        return dict(agent_pcm=None if self.calls == 1 else np.full(1920, .125, dtype='float32'),
                    text_token=self.calls, is_ret=self.calls == 2)

    def decode_text(self, tokens): return 'tokens:' + str(len(tokens))


class TrainFailureRolloutTest(unittest.TestCase):
    def test_four_shards_cover_every_input_once(self):
        rows = [dict(id='case-%04d' % i) for i in reversed(range(1008))]
        groups = [shard(rows, rank, 4) for rank in range(4)]
        self.assertEqual([len(g) for g in groups], [252] * 4)
        self.assertEqual(len({r['id'] for g in groups for r in g}), 1008)
        with self.assertRaises(ValueError): shard(rows[:-1], 0, 4)

    def test_every_frame_token_and_raw_sample_remains_replayable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = np.concatenate((np.array([.25], dtype='float32'), np.zeros(60 * 24000, dtype='float32')))
            wave = root / 'user.wav'
            sf.write(wave, audio, 24000, subtype='FLOAT')
            row = dict(id='train', folder=str(root), input_sha256=hashlib.sha256(wave.read_bytes()).hexdigest(),
                       user_frames=1, input_rate=24000)
            engine = Engine()
            result = collect_case(engine, row, root / 'output', lambda: None)
            trace = [json.loads(line) for line in (root / 'output/trace.jsonl').read_text().splitlines()]
            pcm, rate = sf.read(root / 'output/agent-raw.wav', dtype='float32')
            self.assertEqual(len(trace), 751)
            self.assertEqual(result['native_RET_events'], 1)
            self.assertEqual(result['generated_text'], 'tokens:751')
            self.assertEqual(rate, 24000)
            for frame in trace:
                start = frame['raw_sample_offset']
                self.assertEqual(hashlib.sha256(pcm[start:start + frame['audio_samples']].tobytes()).hexdigest(), frame['pcm_sha256'])
            self.assertFalse(result['live_latency_measured'])
            with self.assertRaises(FileExistsError): collect_case(engine, row, root / 'output', lambda: None)

    def test_changed_or_teacher_contaminated_input_stops_before_native_step(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wave = root / 'user.wav'
            sf.write(wave, np.ones(60 * 24000 + 1, dtype='float32'), 24000, subtype='FLOAT')
            row = dict(id='train', folder=str(root), input_sha256=hashlib.sha256(wave.read_bytes()).hexdigest(), user_frames=1, input_rate=24000)
            engine = Engine()
            with self.assertRaises(ValueError): collect_case(engine, row, root / 'out', lambda: None)
            self.assertEqual(engine.calls, 0)
            self.assertFalse((root / 'out').exists())
