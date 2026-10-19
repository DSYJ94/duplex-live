"""Repair must preserve originals/held-out data and update rebuilt scene metadata."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from data.synthetic.recovery_curriculum import assemble
from evals.moshi_model_download import sha256
from evals.native_v2_source_repair_v2 import repair, INSTRUCTIONS


class RepairPreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source'
        self.source.mkdir()
        self.out = Path(self.temp.name) / 'repair-v2'
        self.scenes = [dict(scenario_id='native2-train-015', split='train', kind='simple',
                           script_family='train-a', user_voice='echo', assistant_voice='marin',
                           utterances=[['user','Wait.'],['assistant','Okay.'],['user','Pause.'],['user','Go.']]),
                       dict(scenario_id='native2-train-017', split='train', kind='simple',
                           script_family='train-b', user_voice='echo', assistant_voice='marin',
                           utterances=[['user','What is the parcel label?'],['assistant','Which project?']])]
        rows = []
        for scene in self.scenes:
            folder = self.source / scene['scenario_id'] / 'utterances'
            folder.mkdir(parents=True)
            receipts = {}
            for i, (role, _) in enumerate(scene['utterances']):
                path = folder / f'{i:02d}_{role}.wav'
                sf.write(path, np.zeros(4800), 24000)
                receipts[i] = dict(request_id='original', sha256=sha256(path))
                path.with_suffix('.receipt.json').write_text(json.dumps(receipts[i]))
            rows.append(assemble(scene, self.source, receipts))
        (self.source/'native2-test-000').mkdir()
        (self.source/'native2-test-000/held.wav').write_bytes(b'heldout original')
        for name in ['quality.jsonl','filter_report.jsonl','specifications.json','generation.json']:
            (self.source/name).write_text('{}')
        (self.source/'manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.findings = [dict(scene=sid, index=i, role='user', text=text,
                             sha256=sha256(self.source/sid/'utterances'/f'{i:02d}_user.wav'))
                         for sid,i,text in [('native2-train-015',3,'Go.'),
                                            ('native2-train-017',0,'What is the parcel label?')]]

    def render(self, text, voice, path, *, instructions):
        self.assertEqual(instructions, INSTRUCTIONS)
        sf.write(path, np.sin(np.arange(12000)*.03)*.1, 24000)
        receipt = dict(request_id='actual-test-receipt', sha256=sha256(path), input_text=text)
        path.with_suffix('.receipt.json').write_text(json.dumps(receipt))
        return receipt

    def run_repair(self, render=None):
        with patch('evals.native_v2_source_repair_v2.diagnose', return_value=(self.scenes,self.findings)), \
             patch('evals.native_v2_source_repair_v2.synthesize', side_effect=render or self.render):
            repair(self.source,self.out)

    def test_originals_unrelated_clips_receipts_and_heldout_are_preserved(self):
        self.run_repair()
        for finding in self.findings:
            p=self.out/'repair-originals'/finding['scene']/f"{finding['index']:02d}_user.wav"
            self.assertEqual(sha256(p),finding['sha256'])
            self.assertEqual(json.loads(p.with_suffix('.receipt.json').read_text())['request_id'],'original')
        self.assertEqual((self.out/'native2-test-000/held.wav').read_bytes(),b'heldout original')
        for row in map(json.loads,(self.out/'manifest.jsonl').read_text().splitlines()):
            pcm,rate=sf.read(self.out/row['scenario_id']/'mixed.wav')
            self.assertEqual(row['duration_ms'],round(len(pcm)/rate*1000))
        self.assertFalse((self.out/'quality.jsonl').exists())
        self.assertFalse(json.loads((self.out/'generation.json').read_text())['verified'])

    def test_heldout_mutation_fails_closed(self):
        def corrupt(*args,**kwargs):
            receipt=self.render(*args,**kwargs)
            (self.out/'native2-test-000/held.wav').write_bytes(b'changed')
            return receipt
        with self.assertRaisesRegex(ValueError,'Unrelated source'):
            self.run_repair(corrupt)

    def test_existing_destination_cannot_be_overwritten(self):
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError,'new versioned directory'):
            self.run_repair()


if __name__ == '__main__':
    unittest.main()
