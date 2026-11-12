import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training.personaplex.matched_evaluation_reuse import pin, materialize, verify, prepared


class ReuseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'old'; self.root.mkdir()
        self.rows = [dict(id='case-%02d'%i, input_sha256='hash-%d'%i, family='ordinary', frames=.08) for i in range(32)]
        self.profile = dict(source='original', trainer_source='trained', seed=42)
        (self.root/'profile.json').write_text(json.dumps(self.profile))
        row=self.rows[0]
        self.case=dict(row, complete=True, input_frames=.08); del self.case['frames']
        folder=self.root/row['id']; folder.mkdir()
        (folder/'result.json').write_text(json.dumps(self.case))
        (folder/'trace.jsonl').write_text(json.dumps(dict(sequence=0,chunk_file='chunk.wav'))+'\n')
        (folder/'chunk.wav').write_bytes(b'original raw PCM fixture')
        (self.root/'results.json').write_text(json.dumps(dict(source='original',required_cases=32,cases=[self.case])))
        partial=self.root/'case-01';partial.mkdir();(partial/'chunk.wav').write_bytes(b'failed partial')

    def test_complete_cases_preserve_bytes_and_partial_is_excluded(self):
        record=pin(self.root,self.rows,self.profile)
        target=Path(self.temp.name)/'new'
        cases=materialize(record,target)
        self.assertEqual(cases,[self.case]);self.assertFalse((target/'case-01').exists())
        self.assertEqual((target/'case-00/chunk.wav').read_bytes(),b'original raw PCM fixture')
        self.assertEqual(json.loads((target/'reuse-origin.json').read_text())['original_source'],'original')
        with self.assertRaises(FileExistsError):materialize(record,target)

    def test_modified_chunk_refuses_reuse(self):
        record=pin(self.root,self.rows,self.profile)
        (self.root/'case-00/chunk.wav').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):verify(self.root,record['files'])

    def test_missing_frame_refuses_reuse(self):
        (self.root/'case-00/trace.jsonl').write_text('')
        with self.assertRaisesRegex(ValueError,'Every original'):pin(self.root,self.rows,self.profile)

    def test_changed_profile_or_input_refuses_reuse(self):
        with self.assertRaisesRegex(ValueError,'profile'):pin(self.root,self.rows,dict(self.profile,seed=1))
        self.rows[0]['input_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'exact native'):pin(self.root,self.rows,self.profile)

    def test_manifest_path_escape_refused(self):
        with self.assertRaisesRegex(ValueError,'aliased'):verify(self.root,{'../outside':dict(bytes=0,sha256='x')})

    def test_subset_cannot_be_called_complete_audit(self):
        (self.root/'report.json').write_text(json.dumps(dict(complete=True,cases=[self.case])))
        self.assertFalse(pin(self.root,self.rows,self.profile)['full_audit'])

    def test_legacy_33_second_case_requires_413_native_frames(self):
        self.rows[0]['frames']=33
        self.case['input_frames']=33
        folder=self.root/'case-00'
        (folder/'result.json').write_text(json.dumps(self.case))
        (self.root/'results.json').write_text(json.dumps(dict(source='original',required_cases=32,cases=[self.case])))
        frames=[]
        for i in range(413):
            name='chunk-%04d.wav'%i
            (folder/name).write_bytes(b'PCM fixture')
            frames.append(json.dumps(dict(sequence=i,chunk_file=name)))
        (folder/'trace.jsonl').write_text('\n'.join(frames)+'\n')
        record=pin(self.root,self.rows,self.profile)
        self.assertEqual(len(record['cases']),1)
        (folder/'trace.jsonl').write_text('\n'.join(frames[:-1])+'\n')
        with self.assertRaisesRegex(ValueError,'Every original native frame'):pin(self.root,self.rows,self.profile)

    def test_prepared_inference_never_invokes_rsync(self):
        record=pin(self.root,self.rows,self.profile)
        target=Path(self.temp.name)/'prepared';materialize(record,target)
        with patch('training.personaplex.matched_evaluation_reuse.subprocess.run',side_effect=FileNotFoundError('rsync')) as run:
            self.assertEqual(prepared(record,target),[self.case]);run.assert_not_called()
        with self.assertRaisesRegex(ValueError,'host-prepared'):prepared(record,Path(self.temp.name)/'absent')
