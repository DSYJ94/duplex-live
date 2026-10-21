"""Source assembly safeguards only; no model or synthetic-label qualification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from data.synthetic.native_control_repair import build, validate
from data.synthetic.native_control_render import load_clip, render, validate_raw
from evals.native_control_coverage import quiet_windows


class ControlSourceRepair(unittest.TestCase):
    def clips(self, root, plan):
        import numpy as np
        import soundfile as sf
        paths = []
        # Explicit fixture signal: never used as speech or a teacher target.
        pcm = (.15 * np.sin(np.arange(4800, dtype=np.float32) * .13)).astype('float32')
        for scene in plan['cases']:
            folder = root / scene['id']
            folder.mkdir(parents=True)
            for i, turn in enumerate(scene['turns']):
                path = folder / f'{i:02d}_{turn["role"]}.wav'
                sf.write(path, pcm, 24000, subtype='FLOAT')
                receipt = dict(source_model='test-fixture-not-speech', input_text=turn['text'],
                               voice=scene['user_voice'] if turn['role'] == 'user' else scene['assistant_voice'],
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                path.with_suffix('.receipt.json').write_text(json.dumps(receipt))
                paths.append(path)
        return paths

    def test_complete_plan_and_no_release_preserve_both_target_groups_without_fake_speech(self):
        plan = build()
        summary = validate(plan)
        self.assertEqual(summary['cases'], 40)
        self.assertEqual(summary['source_turns'], 168)
        for scene in plan['cases']:
            self.assertFalse(scene['training_approved'])
            if scene['release_turn'] is None:
                before = scene['turns'][:scene['quiet_start_after_turn']]
                self.assertTrue(any(t['role'] == 'assistant' for t in before))
                self.assertEqual(scene['turns'][-1]['role'], 'user')

    def test_no_subset_heldout_relabeling_or_quiet_acknowledgment(self):
        plan = build()
        for mutation in ('subset', 'heldout', 'ack', 'nan-gap'):
            changed = copy.deepcopy(plan)
            if mutation == 'subset':
                changed['cases'].pop()
            elif mutation == 'heldout':
                changed['cases'][0]['split'] = 'test'
            elif mutation == 'nan-gap':
                changed['cases'][0]['turns'][0]['gap_before_s'] = float('nan')
            else:
                row = changed['cases'][0]
                row['turns'].append(dict(role='assistant', text='Okay.', gap_before_s=.56))
                row['expected_assistant_turns'].append(3)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(changed)

    def test_actual_complete40_assembly_keeps_quiet_pcm_and_source_bytes(self):
        import soundfile as sf
        import numpy as np
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build()
            path = root / 'plan.json'
            path.write_text(json.dumps(plan))
            files = self.clips(root / 'clips', plan)
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            result = render(path, root / 'clips', root / 'raw')
            self.assertEqual(len(result['cases']), 40)
            self.assertFalse(result['training_approved'])
            self.assertFalse((root / 'raw/train.jsonl').exists())
            for scene, row in zip(plan['cases'], result['cases']):
                pcm, rate = sf.read(root / 'raw' / row['id'] / 'conversation.wav', always_2d=True)
                quiet = row['quiet_contract']
                self.assertEqual(rate, 24000)
                self.assertEqual(np.count_nonzero(pcm[quiet['start_sample']:quiet['end_sample'], 0]), 0)
                if scene['release_turn'] is None:
                    self.assertGreaterEqual((quiet['end_sample'] - quiet['start_sample']) / rate,
                                            scene['trailing_observation_s'])
                self.assertEqual(len(pcm) % 1920, 0)
            self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
            import shutil
            shutil.rmtree(root / 'clips')
            manifest, _ = validate_raw(root / 'raw')
            self.assertEqual(len(manifest['cases']), 40, 'Portable source cannot depend on original Mac clip paths')
            # Even replacing a manifest hash cannot bless PCM absent from the
            # original input clips. Reconstruct the stems independently.
            row = manifest['cases'][0]
            stem = root / 'raw' / row['id'] / 'stem_assistant.wav'
            data, rate = sf.read(stem, dtype='float32')
            data[row['quiet_contract']['start_sample'] + 100] = .05
            sf.write(stem, data, rate, subtype='FLOAT')
            row['artifacts']['stem_assistant.wav'] = hashlib.sha256(stem.read_bytes()).hexdigest()
            (root / 'raw/raw-source.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'differs from immutable original clips'):
                validate_raw(root / 'raw')
            with self.assertRaises(ValueError):
                render(path, root / 'clips', root / 'raw')

    def test_empty_clipped_or_wrong_words_cannot_be_assembled(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build()
            files = self.clips(root / 'clips', plan)
            path = files[0]
            expected = dict(text=plan['cases'][0]['turns'][0]['text'], voice='alloy')
            original = path.read_bytes()
            receipt_path = path.with_suffix('.receipt.json')
            original_receipt = receipt_path.read_text()
            for signal in (np.zeros(4800), np.ones(4800)):
                sf.write(path, signal, 24000, subtype='FLOAT')
                receipt = json.loads(original_receipt)
                receipt['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaises(ValueError):
                    load_clip(path, expected)
            path.write_bytes(original)
            receipt = json.loads(original_receipt)
            receipt['input_text'] = 'Wrong words.'
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                load_clip(path, expected)
            receipt = json.loads(original_receipt)
            receipt['Authorization'] = 'fixture-only-never-a-real-credential'
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'Unexpected receipt fields'):
                load_clip(path, expected)

    def test_missing_last_case_leaves_entire_source_batch_unapproved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build()
            path = root / 'plan.json'
            path.write_text(json.dumps(plan))
            files = self.clips(root / 'clips', plan)
            files[-1].unlink()
            with self.assertRaises(FileNotFoundError):
                render(path, root / 'clips', root / 'raw')
            self.assertFalse((root / 'raw/raw-source.json').exists())
            self.assertTrue((root / 'raw' / plan['cases'][0]['id']).exists())
            with self.assertRaises(ValueError):
                render(path, root / 'clips', root / 'raw')

    def test_control_audio_audit_uses_words_not_a_merged_assistant_event(self):
        sidecar = dict(timeline=dict(meta=dict(quiet_between_user_ordinals=[1, 2]), events=[
            dict(ch='user', kind='utterance', text='First step.', actual_end_ms=1000),
            dict(ch='user', kind='utterance', text='Pause.', actual_end_ms=3000),
            dict(ch='user', kind='utterance', text='Cue.', actual_end_ms=7000),
            dict(ch='assistant', kind='utterance', text='First. Last.', actual_start_ms=1200, actual_end_ms=8500)]),
            alignments=[['First', [1.2, 2.0], 'SPEAKER_MAIN'], ['Last', [8.0, 8.5], 'SPEAKER_MAIN']])
        self.assertEqual(quiet_windows(sidecar)[0]['aligned_words_during_quiet'], [])
        sidecar['alignments'].append(['Okay', [4.0, 4.3], 'SPEAKER_MAIN'])
        self.assertEqual(quiet_windows(sidecar)[0]['aligned_words_during_quiet'], ['Okay'])


if __name__ == '__main__':
    unittest.main()
