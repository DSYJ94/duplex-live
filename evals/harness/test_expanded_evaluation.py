import copy,json,tempfile,unittest,importlib.util,shutil
from pathlib import Path
from training.personaplex.expanded_evaluation import named_adapter,complete_reports,completed
from training.personaplex.matched_evaluation import complete_collection,speech_presence


class Value:
    def numel(self):return 105070592


class ExpandedEvaluation(unittest.TestCase):
    def test_only_full_declared_expanded_adapter_membership_is_accepted(self):
        value=Value();selection=dict(trainer_source='a'*40)
        state=dict(schema='dl-distributed-native-state-v2',step=196,world_size=4,
            identity=dict(source='a'*40,base='personaplex',method='broad-lora64',trainable_parameters=105070592,old_saved_updates_replayed=0,recipe=dict(new_TRAIN=1008,new_DEV=64)),
            training=dict(method='broad-lora64',ownership=dict(trainable_names=['layer.lora_A']),model={'lm.layer.lora_A':value}))
        self.assertEqual(named_adapter(state,selection),{'layer.lora_A':value})
        before=dict(state['training']['model'])
        for bad in (dict(state,step=195),dict(state,world_size=2),dict(state,training=dict(state['training'],model={'other.layer':value})),dict(state,training=dict(state['training'],ownership=dict(trainable_names=[])))):
            with self.assertRaises(ValueError):named_adapter(bad,selection)
        self.assertEqual(state['training']['model'],before)

    def test_live_parent_cannot_enter_model_evaluation(self):
        for state in ('RUNNING','PENDING','FAILED','UNKNOWN'):
            with self.assertRaises(ValueError):completed(Path('/absent'),'a'*40,dict(JobState=state,Account='brics.u6ada'),Path('/backup'))

    def test_completed_parent_requires_all_native_restores_and_full_backup(self):
        import hashlib
        from unittest.mock import patch
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)/'phase';folder=root/'job-123';(folder/'verify').mkdir(parents=True)
            cp=root/'checkpoints/step-0196.pt';cp.parent.mkdir();cp.write_bytes(b'unit common state')
            h=hashlib.sha256(cp.read_bytes()).hexdigest();terminal=dict(training_complete=True,actual_new_step=196,continuation_ready=False,source='trainer',checkpoint=str(cp),checkpoint_sha256=h)
            (folder/'host-terminal.json').write_text(json.dumps(dict(success=True,training=terminal)))
            (folder/'verify/verify-terminal.json').write_text(json.dumps(dict(passed=True,step=196,new_updates=0)))
            rows=[dict(source='trainer',step=196,actual_CUDA=True,native_fresh_restore_passed=True,native_data_qualified=True,all_Adam_RNG_OPAL_native_fields_exact=True,execution='fresh'+str(i),reference_execution='saved'+str(i),pid=100+i)for i in range(4)]
            proof=folder/'verify/restore-all-ranks.json';proof.write_text(json.dumps(dict(passed=True,rows=rows)))
            back=Path(d)/'backup';dest=back/'step-0196';dest.mkdir(parents=True);name=cp.name
            metadata=json.dumps(dict(file=name,step=196,sha256=h)).encode();files=[]
            for n,blob in [(name,cp.read_bytes()),(name+'.sha256.json',metadata)]+[(name+'.rank-%04d.pt'%i,b'unit rank state')for i in range(4)]:
                (dest/n).write_bytes(blob);files.append(dict(file=n,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
            (dest/'backup.json').write_text(json.dumps(dict(source=str(cp),complete=True,step=196,files=files)))
            provider=dict(JobId='123',JobState='COMPLETED',Account='brics.u6ada')
            with patch('training.personaplex.matched_archive.independent_filesystems',return_value=True):
                self.assertEqual(completed(root,'trainer',provider,back)['step'],196)
                rows[-1]['all_Adam_RNG_OPAL_native_fields_exact']=False;proof.write_text(json.dumps(dict(passed=True,rows=rows)))
                with self.assertRaises(ValueError):completed(root,'trainer',provider,back)
                rows[-1]['all_Adam_RNG_OPAL_native_fields_exact']=True;proof.write_text(json.dumps(dict(passed=True,rows=rows)))
                (dest/'backup.json').write_text(json.dumps(dict(source=str(cp),complete=True,step=196,files=files[:-1])))
                with self.assertRaises(ValueError):completed(root,'trainer',provider,back)

    def test_complete64_keeps_legacy32_default_strict(self):
        rows=[dict(id=str(i),input_sha256=str(i))for i in range(64)];value=dict(required_cases=64,cases=[dict(r,complete=True)for r in rows])
        self.assertEqual(len(complete_collection(value,rows,required=64)),64)
        with self.assertRaises(ValueError):complete_collection(value,rows)
        with self.assertRaises(ValueError):complete_collection(dict(value,cases=value['cases'][:-1]),rows,required=64)

    def test_silent_empty_outputs_cannot_pass_new_speech_gate(self):
        self.assertFalse(speech_presence('','',False));self.assertFalse(speech_presence('four','four',False))
        self.assertFalse(speech_presence('four','',True));self.assertTrue(speech_presence('Four.','four',True))

    def reports(self,root):
        for cohort,count in (('control32',32),('dev64',64)):
            for cap in ('baseline','expanded105M'):
                path=root/cohort/('personaplex-'+cap)/'report.json';path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(complete=True,source='source',cases=[dict(id=str(i),input_sha256=str(i))for i in range(count)],optimizer_updates=0,external_API_calls=0,summary={'cases':count})))

    def test_all192_outputs_and_exact_paired_inputs_are_required(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);self.reports(root);r=complete_reports(root,'source')
            self.assertTrue(r['complete192']);self.assertFalse(r['connected_backend_tested']);self.assertTrue(r['semantic_correctness_review_pending'])
        for damage in ('missing','hash','source'):
            with tempfile.TemporaryDirectory()as d:
                root=Path(d);self.reports(root);path=root/'dev64/personaplex-expanded105M/report.json';r=json.loads(path.read_text())
                if damage=='missing':r['cases'].pop()
                elif damage=='hash':r['cases'][0]['input_sha256']='changed'
                else:r['source']='other'
                path.write_text(json.dumps(r))
                with self.assertRaises(ValueError):complete_reports(root,'source')

    def test_archive_ignores_mutable_guard_and_compiler_scratch(self):
        from training.personaplex.expanded_evaluation_archive import static_inventory
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)
            for folder in ('control32','dev64','dev-inputs','job-123/scratch'):(root/folder).mkdir(parents=True)
            (root/'control32/raw.wav').write_bytes(b'raw');(root/'job-123/guard.json').write_text('changing');(root/'job-123/scratch/private').write_text('cache')
            paths={r['path']for r in static_inventory(root)}
            self.assertEqual(paths,{'control32/raw.wav'})

    @unittest.skipUnless(shutil.which('rsync'),'Host archive requires actual rsync')
    def test_full_result_backup_preserves_every_byte_and_never_overwrites(self):
        from unittest.mock import patch
        from training.personaplex.expanded_evaluation_archive import archive
        with tempfile.TemporaryDirectory()as d:
            root=Path(d)/'result';root.mkdir()
            for folder in ('control32','dev64','dev-inputs'):(root/folder).mkdir()
            (root/'control32/raw.wav').write_bytes(b'untouched raw PCM')
            driver=root/'control32/driver';driver.mkdir();(driver/'libcuda.so').symlink_to('/usr/lib/libcuda.so.999')
            (root/'comparison.json').write_text(json.dumps(dict(complete192=True,source='source',optimizer_updates=0)))
            destination=Path(d)/'independent'/'snapshot'
            with patch('training.personaplex.expanded_evaluation_archive.independent_filesystems',return_value=True):
                receipt=archive(root,destination,'source')
                self.assertTrue(receipt['every_file_sha256_verified'])
                self.assertEqual((destination/'control32/raw.wav').read_bytes(),b'untouched raw PCM')
                self.assertTrue((destination/'control32/driver/libcuda.so').is_symlink())
                with self.assertRaises(ValueError):archive(root,destination,'source')

    @unittest.skipUnless(importlib.util.find_spec('numpy')and importlib.util.find_spec('soundfile'),'Native CPU runtime supplies numpy/soundfile')
    def test_dev_audio_contains_only_complete_user_and_fixed_teacher_independent_tail(self):
        import numpy as np
        import hashlib
        from unittest.mock import patch
        from training.personaplex.expanded_evaluation import dev_cases
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);data=root/'source';data.mkdir();rows=[];seen=[]
            for i in range(1072):
                if i<1008:rows.append(dict(id='train'+str(i),fold='TRAIN'));continue
                name='dev'+str(i);path=data/(name+'.wav');path.write_bytes(b'original source');side=path.with_suffix('.json');side.write_text('{}')
                rows.append(dict(id=name,fold='DEV',path=path.name,wav_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sidecar_sha256=hashlib.sha256(side.read_bytes()).hexdigest(),parents=[dict(id=name+'-user',resampled_frames=4)]))
            (data/'manifest.json').write_text(json.dumps(dict(complete1072=True,TRAIN=1008,DEV=64,rows=rows)))
            source=np.array([[.9,.1],[.8,.2],[.7,.3],[.6,.4],[.5,0.]],dtype=np.float32)
            def write(path,pcm,rate,**kwargs):
                self.assertEqual(rate,24000);self.assertEqual(pcm.shape,(4+60*24000,))
                np.testing.assert_array_equal(pcm[:4],source[:4,1]);self.assertFalse(np.any(pcm[4:]))
                self.assertTrue(np.max(pcm)<.5);Path(path).write_bytes(b'only synthetic user');seen.append(Path(path))
            with patch('soundfile.read',return_value=(source,24000)),patch('soundfile.write',side_effect=write):
                cases=dev_cases(data,root/'derived')
            self.assertEqual(len(cases),64);self.assertEqual(len(seen),64)
            self.assertTrue(all(c['teacher_audio_in_model_input']is False for c in cases))


if __name__=='__main__':unittest.main()
