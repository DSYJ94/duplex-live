import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.personaplex.expanded_guard import sha
from training.personaplex.train_failure_host import complete


class TrainFailureHostTest(unittest.TestCase):
    def fixture(self, root):
        rows = [dict(id='train-%04d' % i, input_sha256=str(i), user_frames=1) for i in range(1008)]
        manifest = root / 'input.json'
        manifest.write_text(json.dumps(dict(cases=rows)))
        plan = dict(input_manifest=str(manifest), source='actual-source', selection=dict(step=196))
        for rank in range(4):
            folder = root / ('rank-%02d' % rank)
            folder.mkdir()
            results = []
            for row in rows[rank::4]:
                case = folder / row['id']
                case.mkdir()
                (case / 'agent-raw.wav').write_bytes(b'fixture-not-native-audio')
                (case / 'trace.jsonl').write_bytes(b'fixture-not-native-trace')
                results.append(dict(id=row['id'], input_sha256=row['input_sha256'], complete=True,
                    source_fold='TRAIN', every_saved_PCM_frame_replayed_exact=True,
                    teacher_audio_or_answer_provided=False, backend_connected=False, optimizer_updates=0,
                    raw_wav_sha256=sha(case / 'agent-raw.wav'), trace_sha256=sha(case / 'trace.jsonl'),
                    user_frames=1, response_tail_seconds=60, input_padding_samples=1919,
                    native_frames=751, output_samples=0))
                (case / 'result.json').write_text(json.dumps(results[-1]))
            (folder / 'terminal.json').write_text(json.dumps(dict(complete=True, rank=rank,
                required=252, cases=results, source=plan['source'], actual_CUDA=True,
                input_manifest_sha256='fixture', selection=plan['selection'], optimizer_updates=0,
                learned_RET_head_installed=False, process_identity=dict(pid=rank + 100,
                    pid_namespace='fixture', start_ticks=rank + 1))))
        return plan

    def test_all1008_results_required_and_changed_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch('training.personaplex.train_failure_host.INPUT_SHA', 'fixture'):
            root = Path(temp)
            plan = self.fixture(root)
            result = complete(root, plan)
            self.assertEqual(result['actual_outputs'], 1008)
            self.assertFalse(result['product_accepted'])
            (root / 'rank-03/train-1007/agent-raw.wav').write_bytes(b'changed')
            with self.assertRaises(ValueError): complete(root, plan)

    def test_a_complete_flag_cannot_hide_partial_frame_coverage(self):
        with tempfile.TemporaryDirectory() as temp, patch('training.personaplex.train_failure_host.INPUT_SHA', 'fixture'):
            root = Path(temp)
            plan = self.fixture(root)
            path = root / 'rank-00/terminal.json'
            terminal = json.loads(path.read_text())
            terminal['cases'][0]['native_frames'] -= 1
            path.write_text(json.dumps(terminal))
            with self.assertRaises(ValueError): complete(root, plan)
