import copy
import unittest
from evals.context_native_stream_training import policy,exposure

class NativeStreamTraining(unittest.TestCase):
    def declared(self):
        return policy(dict(cases=[dict(teacher=dict(id=str(i),kind=('private' if i<27 else 'simple' if i<39 else 'clarify' if i<44 else 'cancel'))) for i in range(49)]))
    def steps(self,p):
        return [dict(step=i+1,native_stream_teacher_microbatch=[dict(id=r['id'],kind=r['kind'],weight=r['weight_when_selected'],valid_targets=2,gradient_calls=2,prefix_span_user_audio_loss_targets=0) for r in p['teachers'][i::8]]) for i in range(8)]
    def test_all49_once_with_balanced_kind_epoch_mass_and_original_terms_retained(self):
        p=self.declared();self.assertTrue(p['all_original154_terms_and_losses_unchanged'])
        rows=self.steps(p);self.assertTrue(exposure(rows,p))
        self.assertEqual(len({r['id'] for s in rows for r in s['native_stream_teacher_microbatch']}),49)
        for kind in ('private','simple','clarify','cancel'):
            self.assertAlmostEqual(sum(r['weight_per_epoch'] for r in p['teachers'] if r['kind']==kind),.25)
        self.assertFalse(p['sampler_or_dtype_or_tf32_changed'])
    def test_subset_duplicate_wrong_weight_or_nonzero_excluded_targets_fail(self):
        p=self.declared();rows=self.steps(p)
        for field,value in [('id','unapproved'),('weight',.5),('prefix_span_user_audio_loss_targets',1),('gradient_calls',0)]:
            bad=copy.deepcopy(rows);bad[0]['native_stream_teacher_microbatch'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):exposure(bad,p)
        bad=copy.deepcopy(rows);bad[7]['native_stream_teacher_microbatch'].pop()
        with self.assertRaises(ValueError):exposure(bad,p)
    def test_native_readiness_cannot_replace_complete154_preflight(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from evals import native_audio49_chain as chain
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);folder=root/'runs/native-audio49-training-preflight-1';folder.mkdir(parents=True)
            p=dict(mode='native520-complete154-native-stream49-auxiliary-v1')
            complete=dict(passed=True,optimizer_updates=0,total_components=153,all_parameters_unchanged=True,source_commit='source',policy=p)
            path=folder/'complete-balanced-preflight.json';path.write_text(json.dumps(complete))
            pre=dict(passed=True,optimizer_updates=0,all_parameters_unchanged=True,source_commit='source',cases=[1,2,3,4],policy=p,complete_balanced_preflight_sha256=chain.digest(path))
            (folder/'audio49-bf16-preflight.json').write_text(json.dumps(pre))
            with patch.object(chain,'BASE',root),patch.object(chain,'validate'),patch.object(chain,'call') as call:
                with self.assertRaises(AssertionError):chain.submit(root/'receipt',dict(jobs={'readiness':'1'},source='source',native_stream_auxiliary=True),'submit-training','1')
                call.assert_not_called()
    def test_missing_teacher_or_kind_cannot_change_inventory(self):
        for cases in [[dict(teacher=dict(id=str(i),kind='private')) for i in range(49)],[],[dict(teacher=dict(id='duplicate',kind='private')) for i in range(49)]]:
            with self.assertRaises(ValueError):policy(dict(cases=cases))
