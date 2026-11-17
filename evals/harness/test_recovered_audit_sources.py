import unittest
from training.personaplex.matched_evaluation import audit_sources


class RecoveredAuditSourceTest(unittest.TestCase):
    def test_default_source_contract_unchanged(self):
        self.assertEqual(audit_sources(dict(source='old',trainer_source='trainer'),'old','trainer'),dict(source='old'))
        with self.assertRaises(ValueError):audit_sources(dict(source='old',trainer_source='trainer'),'new','trainer')

    def test_reused_generation_and_new_audit_are_both_explicit(self):
        result=audit_sources(dict(source='a'*40,trainer_source='trainer'),'b'*40,'trainer','a'*40)
        self.assertEqual(result,dict(source='a'*40,generation_source='a'*40,audit_source='b'*40))
        with self.assertRaises(ValueError):audit_sources(dict(source='c'*40,trainer_source='trainer'),'b'*40,'trainer','a'*40)
        with self.assertRaises(ValueError):audit_sources(dict(source='a'*40,trainer_source='other'),'b'*40,'trainer','a'*40)
        with self.assertRaises(ValueError):audit_sources(dict(source='old',trainer_source='trainer'),'new','trainer','old')

    def test_real_audit_reuses_all32_fixture_waves_and_records_both_sources(self):
        import json,tempfile
        from pathlib import Path
        from types import SimpleNamespace
        import numpy as np
        import soundfile as sf
        from training.personaplex.matched_evaluation import audit
        from training.personaplex.matched_guard import sha
        with tempfile.TemporaryDirectory()as temporary:
            root=Path(temporary);out=root/'personaplex-expanded105M';out.mkdir();rows=[];cases=[];original={}
            for i in range(32):
                ident='fixture-%02d'%i;folder=out/ident;folder.mkdir()
                for name in ('derived-user-24k.wav','chunk-0000.wav'):
                    path=folder/name;sf.write(path,np.ones(1920,dtype=np.float32)*.1,24000,subtype='FLOAT');original[path]=sha(path)
                (folder/'trace.jsonl').write_text(json.dumps(dict(sequence=0,chunk_file='chunk-0000.wav',audio_samples=1920))+'\n')
                rows.append(dict(id=ident,input_sha256='input-'+str(i)))
                cases.append(dict(id=ident,input_sha256='input-'+str(i),complete=True,family='ordinary',generated_text='hello'))
            (out/'results.json').write_text(json.dumps(dict(required_cases=32,cases=cases)))
            (out/'profile.json').write_text(json.dumps(dict(source='a'*40,trainer_source='c'*40)))
            args=SimpleNamespace(root=root,base='personaplex',capacity='expanded105M',source='b'*40,trainer_source='c'*40,lease='fixture')
            audit(args,rows,required=32,guard=lambda *a:None,recognizer_factory=lambda:lambda *a,**k:dict(text='hello'),generation_source='a'*40,require_spoken_content=True)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['source'],'a'*40);self.assertEqual(report['audit_source'],'b'*40)
            self.assertEqual(report['summary']['word_pass'],32);self.assertTrue(report['complete'])
            self.assertTrue(all(sha(path)==digest for path,digest in original.items()))
