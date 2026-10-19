import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evals import native_fresh_source_reapproval as gate


class DerivedSourceApprovalTests(unittest.TestCase):
    def test_authorization_must_match_committed_owner_scope(self):
        self.assertEqual(gate.authorization()['owner_hearing'],'room sign')
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'authorization.json'
            altered=gate.authorization();altered['owner_scope_approval']='not approved'
            p.write_text(json.dumps(altered))
            with patch.object(gate,'AUTHORIZATION',p),self.assertRaises(ValueError):gate.authorization()

    def test_replay_cannot_grant_training_or_native_behavior_approval(self):
        with patch.object(gate,'recompute',return_value=dict(all20_diagnostic_meets_proposed_bars=True)):
            q=gate.quality(Path('data'),Path('primary'),Path('diagnostic'))
        self.assertTrue(q['input_quality_approved'])
        for key in ('training_approved','inference_approved','room_rune_equivalent','passing_subset_accepted'):
            self.assertFalse(q[key])
        self.assertTrue(q['independent_full10_candidate_review_required'])

    def test_stored_approval_recomputed_and_header_subset_or_audio_tampering_rejected(self):
        q=dict(schema='dl-native-fresh2-derived-quality-v1',profile=gate.PROFILE,passed=True,
               primary_quality_path='primary',turbo_diagnostic_path='turbo',
               evidence=dict(results=[dict(scenario_id=str(i),wav_sha256='original') for i in range(20)]))
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'quality.json'
            with patch.object(gate,'quality',return_value=q),patch.object(gate,'load_inputs',return_value='all20'):
                p.write_text(json.dumps(q));self.assertEqual(gate.approved_inputs(Path(root),p),'all20')
                variants=[]
                for key,value in [('profile','diagnostic-only'),('inference_approved',True),('passed',False)]:
                    changed=copy.deepcopy(q);changed[key]=value;variants.append(changed)
                changed=copy.deepcopy(q);changed['evidence']['results'].pop();variants.append(changed)
                changed=copy.deepcopy(q);changed['evidence']['results'][0]['wav_sha256']='changed';variants.append(changed)
                for changed in variants:
                    p.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):gate.approved_inputs(Path(root),p)

    def test_any_failed_clip_keeps_complete_failed_report_and_prevents_approval(self):
        q=dict(passed=False,evidence=dict(complete_cases=20,complete_clips=38))
        with tempfile.TemporaryDirectory() as root,patch.object(gate,'quality',return_value=q), \
             patch.object(gate,'approved_inputs') as reader:
            out=Path(root)/'new-version'
            with self.assertRaises(RuntimeError):gate.run(Path('data'),Path('primary'),Path('turbo'),out)
            self.assertEqual(json.loads((out/'quality.json').read_text()),q);reader.assert_not_called()


if __name__=='__main__':unittest.main()
