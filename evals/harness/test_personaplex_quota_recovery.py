import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from training.personaplex import matched_interrupted as recovery,matched_continue_policy as policy
from training.personaplex.matched_reuse import ORIGIN,DATA_SHA
from evals.harness.test_personaplex_matched_training_only import FreshRestoreBinding


class InterruptedQuotaRecovery(unittest.TestCase):
    def fixture(self,root):
        saved=root/'preserved/personaplex-narrow-step3';saved.mkdir(parents=True)
        row,original=FreshRestoreBinding().fixture(saved)
        old=Path(row['checkpoint']);checkpoint=saved/'interrupted.pt';old.rename(checkpoint)
        Path(str(old)+'.native-reference.json').rename(Path(str(checkpoint)+'.native-reference.json'))
        row.update(base='personaplex',capacity='narrow',checkpoint=str(checkpoint),checkpoint_sha256=recovery.STATE_SHA,
            quota_interrupted_state='dl-moshi-quota-interrupted-v1')
        reference=dict(source=recovery.PREDECESSOR,checkpoint_sha256=recovery.STATE_SHA,step=3,execution_id=recovery.EXECUTION)
        Path(str(checkpoint)+'.native-reference.json').write_text(json.dumps(reference))
        record=dict(source=recovery.PREDECESSOR,completed=False,actual_new_updates=3,total_lineage_updates=3,checkpoint_sha256=recovery.STATE_SHA)
        (saved/'training-interrupted.json').write_text(json.dumps(record))
        (saved/'training.jsonl').write_text('\n'.join(json.dumps(dict(source=recovery.PREDECESSOR,step=i,objective=10.,LR=1e-7)) for i in (1,2,3)))
        failed=root/'preserved/failure-8235976';failed.mkdir()
        (failed/'actual-failure.json').write_text(json.dumps(dict(job_id='8235976',state='FAILED',exit_code='1:0',elapsed_seconds=644,GPUs=2,guard_error='EDQUOT')))
        path=Path(row['restore']);v=json.loads(path.read_text());v.update(base='personaplex',capacity='narrow',checkpoint_sha256=recovery.STATE_SHA,restored_training_step=3,reference_execution_id=recovery.EXECUTION);path.write_text(json.dumps(v));row['restore_sha256']=policy.g.sha(path)
        path=Path(row['restore_terminal']);v=json.loads(path.read_text());v.update(base='personaplex',capacity='narrow');path.write_text(json.dumps(v))
        return row,original,reference,saved

    def test_real_failed_interruption_and_new_exact_restore_reduce_only_saved_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row,original,reference,saved=self.fixture(root)
            with patch.object(recovery,'DURABLE',root),patch.object(recovery,'PRESERVED',saved),patch.object(recovery,'authority',return_value='actual-owner-grant'):
                self.assertEqual(policy.restored_entry(row,'new',DATA_SHA,original),3)
                failed=root/'preserved/failure-8235976/actual-failure.json';v=json.loads(failed.read_text());v['state']='COMPLETED';failed.write_text(json.dumps(v))
                with self.assertRaisesRegex(ValueError,'remain recorded as failed'):policy.restored_entry(row,'new',DATA_SHA,original)

    def test_unsaved_context_state_arbitrary_state_and_missing_new_native_proof_rejected(self):
        for field,value in [('base','context-spanning'),('capacity','broad'),('checkpoint_sha256','different')]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);row,old,reference,saved=self.fixture(root);row[field]=value
                with patch.object(recovery,'DURABLE',root),patch.object(recovery,'PRESERVED',saved),patch.object(recovery,'authority',return_value='actual-owner-grant'):
                    with self.assertRaises(ValueError):policy.restored_entry(row,'new',DATA_SHA,old)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);row,old,reference,saved=self.fixture(root)
            v=json.loads(Path(row['restore']).read_text());v['native_logits_exact']=False;Path(row['restore']).write_text(json.dumps(v));row['restore_sha256']=policy.g.sha(row['restore'])
            with patch.object(recovery,'DURABLE',root),patch.object(recovery,'PRESERVED',saved),patch.object(recovery,'authority',return_value='actual-owner-grant'):
                with self.assertRaises(ValueError):policy.restored_entry(row,'new',DATA_SHA,old)

    def test_previous_producer_is_allowed_only_for_exact_saved_model_state(self):
        a=SimpleNamespace(source='new',base='personaplex',capacity='narrow',checkpoint_sha=recovery.STATE_SHA,checkpoint=str(recovery.PRESERVED/'interrupted.pt'))
        with patch.object(recovery,'authority',return_value='actual-owner-grant'):
            self.assertTrue(recovery.reference_allowed(a,recovery.PREDECESSOR))
            a.capacity='broad';self.assertFalse(recovery.reference_allowed(a,recovery.PREDECESSOR))
            self.assertFalse(recovery.reference_allowed(a,'invented-producer'))


if __name__=='__main__':unittest.main()
