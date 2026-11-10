import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from training.personaplex import matched_normal88 as n,matched_guard as g,matched_identity as identity,matched_native as native


class Normal88Scope(unittest.TestCase):
    def test_only_exact_approved_rollback_state_and_replay_are_allowed(self):
        a=SimpleNamespace(base='personaplex',capacity='narrow',checkpoint_sha=n.SHA,checkpoint=str(n.CHECKPOINT))
        self.assertTrue(n.reference_allowed(a,n.PREDECESSOR))
        a.checkpoint_sha='different';self.assertFalse(n.reference_allowed(a,n.PREDECESSOR))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'grant.json';value=json.loads(n.AUTH.read_text());value['deliberate_saved_update_replay']=[89,90];p.write_text(json.dumps(value))
            with patch.object(n,'AUTH',p),self.assertRaises(ValueError):n.authority()

    def test_normal_checkpoint_reference_cannot_be_rewritten_or_step_changed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=root/'preserved/live.pt';state.parent.mkdir();old=root/'runs/quota-recovery-8260542/personaplex/narrow/train';old.mkdir(parents=True)
            ref=dict(source=n.PREDECESSOR,step=88,checkpoint_sha256=n.SHA,execution_id='old-original')
            original=json.loads(n.PROOF.read_text());original['normal88'].update(path=str(state),execution_id='old-original')
            proof=root/'proof.json';proof.write_text(json.dumps(original));(root/'failure8273944-normal88-preservation.json').write_text(json.dumps(original));(old/'live.pt.native-reference.json').write_text(json.dumps(ref))
            row=dict(base='personaplex',capacity='narrow',checkpoint=str(state),checkpoint_sha256=n.SHA,normal88_recovery=n.MARKER)
            with patch.object(n,'DURABLE',root),patch.object(n,'CHECKPOINT',state),patch.object(n,'PROOF',proof):
                self.assertTrue(n.validate(row,ref,88))
                with self.assertRaises(ValueError):n.validate(row,dict(ref,step=89),89)
                (old/'live.pt.native-reference.json').write_text(json.dumps(dict(ref,execution_id='rewritten')))
                with self.assertRaisesRegex(ValueError,'not be rewritten'):n.validate(row,ref,88)

    def test_actual_restore_comparison_stays_exact_and_retains_mismatch_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=SimpleNamespace(source='actual-new',base='personaplex',capacity='narrow',checkpoint_sha='state-sha')
            expected={'reference':{'hidden':'old-hidden','text_logits':'old-logits'}}
            with patch.object(native,'fingerprint',return_value=expected):
                self.assertTrue(identity.compare_fingerprints(a,None,None,dict(fingerprints=expected),root))
            other=root/'other';other.mkdir()
            actual={'reference':{'hidden':'changed','text_logits':'old-logits'}}
            with patch.object(native,'fingerprint',return_value=actual):
                self.assertFalse(identity.compare_fingerprints(a,None,None,dict(fingerprints=expected),other))
            recorded=json.loads((other/'native-restore-comparison.json').read_text())
            self.assertFalse(recorded['passed']);self.assertEqual(recorded['actual'],actual);self.assertEqual(recorded['expected'],expected)


if __name__=='__main__':unittest.main()
