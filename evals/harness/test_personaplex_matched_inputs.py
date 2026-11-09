import json
from pathlib import Path
import tempfile
import unittest
import os
from training.personaplex.matched_inputs import component_teacher_caches,sha
try:
    import torch
except ImportError:
    torch=None


class OriginalManifestBindings(unittest.TestCase):
    def test_compact_audit_without_teacher_uses_exact_original_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);path=folder/'actual-future.json'
            cache='/projects/u6si/duplex-live/runs/recovery-codes-original'
            path.write_text(json.dumps(dict(teacher=dict(cache=cache))))
            audit=dict(component=str(folder),manifest_sha256=sha(path),passed=True)
            self.assertEqual(component_teacher_caches([audit]),{cache})

    def test_action_only_component_remains_eligible_without_inventing_teacher(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);path=folder/'action-future.json'
            path.write_text(json.dumps(dict(target_kind='cancel_request')))
            self.assertEqual(component_teacher_caches([dict(component=str(folder),manifest_sha256=sha(path))]),set())

    def test_changed_manifest_is_rejected_before_its_paths_are_used(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);path=folder/'early-future.json';path.write_text('{}')
            row=dict(component=str(folder),manifest_sha256=sha(path));path.write_text('{"teacher": {}}')
            with self.assertRaisesRegex(ValueError,'manifest changed'):component_teacher_caches([row])


@unittest.skipUnless(torch is not None,'Torch CPU runtime required')
class OriginalVerifierConsumer(unittest.TestCase):
    def fixture(self,root):
        from training.personaplex import matched_inputs as m
        from training.personaplex.matched_legacy import LEGACY_SOURCE,LEGACY_TEACHER_SHA
        from evals.context_native_audio49_data import digest
        groups={};records={}
        for name,count in [('control40',40),('reasoning32',32)]:
            engine=root/(name+'-engine');cache=root/(name+'-cache');engine.mkdir();cache.mkdir()
            (cache/'cache.json').write_text('{}');rows=[];original=[]
            for i in range(count):
                ident=name+':'+str(i);path=engine/ident/'inputs.pt';path.parent.mkdir()
                saved=dict(inputs=torch.zeros(1,17,3,dtype=torch.int64),mask=torch.ones(3,dtype=torch.bool),
                    targets=torch.tensor([3,5,4]),codes=torch.ones(17,3,dtype=torch.int64))
                torch.save(saved,path)
                row=dict(id=ident,kind='ordinary',fold='TRAIN',inputs=saved['inputs'],mask=saved['mask'],
                    targets=saved['targets'],codes=saved['codes'][None],audio_mask=saved['mask'][None],
                    calls=[],source_sha256=sha(path))
                rows.append(row);original.append(dict(id=ident,file='inputs.pt',family='ordinary',calls=[]))
            (engine/'engine-inputs.json').write_text(json.dumps(dict(cases=original)))
            groups[name]=rows;records[name]=dict(engine=str(engine),cache=str(cache),complete=True,
                engine_report_sha256=sha(engine/'engine-inputs.json'),cache_report_sha256=sha(cache/'cache.json'),
                tensor_fingerprints=[dict(id=r['id'],tensors={k:digest(v) for k,v in r.items() if isinstance(v,torch.Tensor)}) for r in rows])
        legacy=root/'legacy';legacy.mkdir();torch.save(groups,legacy/'inputs.pt')
        report=dict(schema='dl-moshi-original-input-verifier-v1',source='a'*40,legacy_source=LEGACY_SOURCE,
            legacy_teacher_sha256=LEGACY_TEACHER_SHA,wrapper_sha256=sha(Path(m.__file__).with_name('matched_legacy.py')),
            passed=True,complete40_and32=True,actual_CPU_only=True,pid=os.getpid()+100,
            old_thresholds_audio_words_receipts_and_folds_unchanged=True,optimizer_updates=0,
            native_gradient_restore_or_behavior_approved=False,inputs_sha256=sha(legacy/'inputs.pt'),cases=records)
        (legacy/'report.json').write_text(json.dumps(report))
        return m,groups,records,legacy,report

    def test_complete_both_original_cohorts_bind_every_original_tensor(self):
        with tempfile.TemporaryDirectory() as root:
            m,groups,records,legacy,_=self.fixture(Path(root))
            for name,count in [('control40',40),('reasoning32',32)]:
                r=records[name];rows=m.engine_rows(Path(r['cache']),Path(r['engine']),legacy,'a'*40,reasoning=name=='reasoning32')
                self.assertEqual(len(rows),count)
                self.assertTrue(torch.equal(rows[-1]['inputs'],groups[name][-1]['inputs']))

    def test_report_flag_cannot_hide_changed_artifact_or_evaluation_fold(self):
        with tempfile.TemporaryDirectory() as root:
            m,groups,records,legacy,report=self.fixture(Path(root));r=records['control40']
            groups['control40'][0]['fold']='DEV';torch.save(groups,legacy/'inputs.pt')
            report['inputs_sha256']=sha(legacy/'inputs.pt');(legacy/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'TRAIN input cohort'):m.engine_rows(Path(r['cache']),Path(r['engine']),legacy,'a'*40)
            groups['control40'][0]['fold']='TRAIN';groups['control40'][0]['source_sha256']='0'*64
            torch.save(groups,legacy/'inputs.pt');report['inputs_sha256']=sha(legacy/'inputs.pt');(legacy/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'artifact changed'):m.engine_rows(Path(r['cache']),Path(r['engine']),legacy,'a'*40)


if __name__=='__main__':unittest.main()
