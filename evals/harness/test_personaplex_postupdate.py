import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex import matched_failed_state as f,matched_continue_policy as p,matched_guard as g
from training.personaplex.matched_reuse import DATA_SHA
from evals.harness.test_personaplex_matched_training_only import FreshRestoreBinding


class FailedStateRecovery(unittest.TestCase):
    def fixture(self,root):
        saved=root/'saved';folder=saved/'personaplex';folder.mkdir(parents=True)
        row,original=FreshRestoreBinding().fixture(folder)
        old=Path(row['checkpoint']);checkpoint=folder/'interrupted.pt';old.rename(checkpoint)
        Path(str(old)+'.native-reference.json').rename(Path(str(checkpoint)+'.native-reference.json'))
        digest=f.STATES['personaplex'][1];execution='original-real-execution'
        reference=dict(source=f.PREDECESSOR,step=89,checkpoint_sha256=digest,execution_id=execution)
        Path(str(checkpoint)+'.native-reference.json').write_text(json.dumps(reference))
        row.update(base='personaplex',capacity='narrow',checkpoint=str(checkpoint),checkpoint_sha256=digest,postupdate_failure_recovery=f.MARKER)
        (folder/'training.jsonl').write_text('\n'.join(json.dumps(dict(source=f.PREDECESSOR,step=i,objective=4.,LR=1e-7)) for i in range(4,90)))
        (folder/'training-interrupted.json').write_text(json.dumps(dict(source=f.PREDECESSOR,completed=False,total_lineage_updates=89,actual_new_updates=86,checkpoint_sha256=digest)))
        evidence=json.loads(f.EVIDENCE.read_text())
        entry=next(r for r in evidence['lineages'] if r['base']=='personaplex')
        entry.update(preserved_checkpoint=str(checkpoint),reference_execution_id=execution,
            inventory={x.name:dict(sha256=g.sha(x),bytes=x.stat().st_size) for x in folder.iterdir() if x.name in ('interrupted.pt.native-reference.json','training.jsonl','training-interrupted.json')})
        manifest=root/'evidence.json';manifest.write_text(json.dumps(evidence));(saved/'actual-failure-and-state.json').write_text(json.dumps(evidence))
        proof=Path(row['restore']);v=json.loads(proof.read_text());v.update(base='personaplex',capacity='narrow',checkpoint_sha256=digest,restored_training_step=89,reference_execution_id=execution);proof.write_text(json.dumps(v));row['restore_sha256']=g.sha(proof)
        terminal=Path(row['restore_terminal']);v=json.loads(terminal.read_text());v.update(base='personaplex',capacity='narrow');terminal.write_text(json.dumps(v))
        return row,original,saved,manifest,reference

    def test_actual_failed_state_can_resume_only_after_new_full_native_restore(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row,old,saved,manifest,reference=self.fixture(root)
            with patch.object(f,'SAVED',saved),patch.object(f,'EVIDENCE',manifest),patch.object(f,'authority',return_value='actual-grant'):
                self.assertEqual(p.restored_entry(row,'new',DATA_SHA,old),89)
                proof=Path(row['restore']);v=json.loads(proof.read_text());v['native_logits_exact']=False;proof.write_text(json.dumps(v));row['restore_sha256']=g.sha(proof)
                with self.assertRaisesRegex(ValueError,'distinct actual successful'):p.restored_entry(row,'new',DATA_SHA,old)

    def test_wrong_state_or_false_success_cannot_reduce_remaining_work(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row,old,saved,manifest,reference=self.fixture(root)
            with patch.object(f,'SAVED',saved),patch.object(f,'EVIDENCE',manifest),patch.object(f,'authority',return_value='actual-grant'):
                for field,value in [('capacity','broad'),('checkpoint_sha256','other'),('postupdate_failure_recovery','wrong')]:
                    bad=dict(row);bad[field]=value
                    with self.assertRaises(ValueError):f.validate_failed(bad,reference,89)
                bad=dict(reference,step=90)
                with self.assertRaises(ValueError):f.validate_failed(row,bad,90)
                value=json.loads(manifest.read_text());value['actual_state']='COMPLETED'
                manifest.write_text(json.dumps(value));(saved/'actual-failure-and-state.json').write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'recorded as failed'):f.validate_failed(row,reference,89)

    def test_saved_logs_are_immutable_and_do_not_claim_unsaved_context_updates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);row,old,saved,manifest,reference=self.fixture(root)
            with patch.object(f,'SAVED',saved),patch.object(f,'EVIDENCE',manifest),patch.object(f,'authority',return_value='actual-grant'):
                (saved/'personaplex/training.jsonl').write_text('invented progress')
                with self.assertRaisesRegex(ValueError,'receipt/log changed'):f.validate_failed(row,reference,89)
        self.assertEqual(f.STATES['context-spanning'][0],80)

    def test_old_start_limit_does_not_expand_without_exact_new_grant(self):
        lease=dict(continuation_authorization_sha256=g.sha(g.EXTENSION),continuation_interval=6)
        with self.assertRaises(ValueError):g.phase_limit(lease)
        lease['postupdate_recovery_authorization_sha256']=f.authority()
        self.assertEqual(g.phase_limit(lease),112*3600)
        with self.assertRaises(ValueError):g.phase_limit(dict(lease,continuation_interval=7))
        with self.assertRaises(ValueError):g.phase_limit(dict(lease,postupdate_recovery_authorization_sha256='wrong'))


if __name__=='__main__':unittest.main()
