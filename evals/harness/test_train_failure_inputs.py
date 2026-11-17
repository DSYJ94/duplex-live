import copy
import json
import unittest
from training.personaplex.train_failure_inputs import build,RECORD_KEYS


class TrainFailureInputsTest(unittest.TestCase):
    def fixture(self):
        rows = []
        records = []
        for i in range(1072):
            ident = 'source-%04d'%i
            fold = 'TRAIN' if i < 1008 else 'DEV'
            rows.append(dict(id=ident,fold=fold,path='scenes/'+ident+'.wav',wav_sha256='a'*64,
                             sidecar_sha256='b'*64,parents=[dict(id=ident+'-user',resampled_frames=24000)]))
            records.append(dict(id=ident,fold=fold,question='Question '+str(i),response='GOLD-DO-NOT-EXPORT',numeric_target='GOLD-TARGET'))
        return dict(complete1072=True,TRAIN=1008,DEV=64,rows=rows),dict(exact_semantic_records=records)

    def test_complete_inputs_exclude_gold_and_all_dev(self):
        manifest,semantic = self.fixture()
        value = build(manifest,semantic)
        self.assertEqual(len(value['records']),1008)
        self.assertTrue(all(set(r)==RECORD_KEYS and r['fold']=='TRAIN' for r in value['records']))
        self.assertNotIn('GOLD-',json.dumps(value))
        self.assertFalse(value['source_wave_hashes_rechecked'])
        self.assertFalse(value['native_inference_started'])
        self.assertTrue(all(r['user_channel']==1 and r['user_frames']==24000 for r in value['records']))

    def test_fold_or_membership_corruption_cannot_turn_dev_into_train(self):
        for kind in ('fold','duplicate','subset'):
            manifest,semantic = self.fixture()
            if kind == 'fold':semantic['exact_semantic_records'][-1]['fold']='TRAIN'
            if kind == 'duplicate':manifest['rows'][1]['id']=manifest['rows'][0]['id']
            if kind == 'subset':manifest['rows'].pop()
            with self.assertRaises(ValueError):build(manifest,semantic)

    def test_invalid_user_geometry_paths_and_hashes_fail(self):
        for kind in ('frames','path','hash','parent'):
            manifest,semantic = self.fixture()
            row = manifest['rows'][0]
            if kind == 'frames':row['parents'][0]['resampled_frames']=False
            if kind == 'path':row['path']='../other.wav'
            if kind == 'hash':row['wav_sha256']='changed'
            if kind == 'parent':row['parents'].append(copy.deepcopy(row['parents'][0]))
            with self.assertRaises(ValueError):build(manifest,semantic)
