"""Artificial durable responses must retain all168 checks, including real errors."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.synthetic.native_control_repair import build
from data.synthetic.native_control_render import render, sha256
from evals.harness import test_native_control_repair as source_fixtures
from evals.native_control_source_quality import ASR_WEIGHT, assess_clip
from evals.native_control_source_reaudit import reassess
from data.synthetic.filter import norm_words


class DurableSourceReaudit(unittest.TestCase):
    def test_complete_numeric_reassessment_keeps_genuine_word_failure(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build()
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            source_fixtures.ControlSourceRepair().clips(root / 'clips', plan)
            raw = root / 'raw'
            manifest = render(plan_path, root / 'clips', raw)
            parent_root = root / 'old-pipeline/source-audit'
            parent_root.mkdir(parents=True)
            (parent_root.parent / 'pipeline.json').write_text(json.dumps(dict(
                source_commit='fixture-parent', phase='stopped_preserve_all_artifacts', failed_phase='verify_source168')))
            results = []
            for scene in plan['cases']:
                clips = []
                for i, turn in enumerate(scene['turns']):
                    tokens = norm_words(turn['text'])
                    response = dict(text=turn['text'], segments=[dict(words=[
                        dict(text=t, start=.01 + n * .17 / len(tokens), end=.01 + (n + .8) * .17 / len(tokens))
                        for n, t in enumerate(tokens)])])
                    name = f'{scene["id"]}-{i:02d}-asr.json'
                    (parent_root / name).write_text(json.dumps(response))
                    wav = raw / scene['id'] / 'utterances' / f'{i:02d}_{turn["role"]}.wav'
                    pcm, rate = sf.read(wav, dtype='float32')
                    verdict = assess_clip(turn['text'], response, pcm, rate)
                    self.assertTrue(verdict['passed'])
                    verdict.update(passed=False, reasons=['Artificial representation-only failure'])
                    clips.append(dict(turn_index=i, role=turn['role'], wav_sha256=sha256(wav),
                                      receipt_sha256=sha256(wav.with_suffix('.receipt.json')),
                                      asr_file=name, asr_sha256=sha256(parent_root / name), **verdict))
                results.append(dict(id=scene['id'], split='train', passed=False, clips=clips))
            report = dict(schema='dl-native-control3-source-word-audit-v1',
                          code_sha256=hashlib.sha256(b'fixture-source').hexdigest(),
                          renderer_sha256=sha256(Path('data/synthetic/native_control_render.py')),
                          passed=False, all40_complete=True, complete_clips=168,
                          training_approved=False, actual_teacher_collection_completed=False,
                          raw_manifest_sha256=sha256(raw / 'raw-source.json'), plan_sha256=manifest['plan_sha256'],
                          asr_model='whisper-turbo', asr_weight_sha256=ASR_WEIGHT,
                          expected_text_prompt_used=False, original_waveforms_changed=False,
                          passing_subset_accepted=False, wer_bar=.05, peak_bar=.995, results=results)
            parent = parent_root / 'source-word-audit.json'
            parent.write_text(json.dumps(report))
            with patch('evals.native_control_source_reaudit.subprocess.check_output', return_value=b'fixture-source'):
                reviewed = reassess(raw, parent, root / 'derived-good')
                self.assertTrue(reviewed['passed'])
                self.assertEqual(reviewed['complete_clips'], 168)
                self.assertFalse(reviewed['training_approved'])
                self.assertEqual(sha256(parent_root / results[0]['clips'][0]['asr_file']),
                                 sha256(root / 'derived-good' / results[0]['clips'][0]['asr_file']))
                # A genuine changed word must still fail after the same numerical
                # representation repair. This is a new artificial fixture version.
                clip = report['results'][0]['clips'][0]
                response_path = parent_root / clip['asr_file']
                response = json.loads(response_path.read_text())
                response['text'] = 'Wrong ' + response['text']
                response_path.write_text(json.dumps(response))
                clip['asr_sha256'] = sha256(response_path)
                parent.write_text(json.dumps(report))
                failed = reassess(raw, parent, root / 'derived-failed')
                self.assertFalse(failed['passed'])
                self.assertEqual(len(failed['results']), 40)
                self.assertFalse(failed['results'][0]['passed'])


if __name__ == '__main__':
    unittest.main()
