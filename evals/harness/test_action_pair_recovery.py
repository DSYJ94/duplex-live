import tempfile,unittest
from pathlib import Path
from training.personaplex.action_pair_recovery import clone_raw,execute_recovery
from training.personaplex.evaluation_result_archive import inventory
from training.personaplex import expanded_guard as g


class PairRecoveryTest(unittest.TestCase):
    def test_clone_reuses_verified_bytes_without_overwriting_source(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);source=root/'old';path=source/'control32/model/case/chunk.wav';path.parent.mkdir(parents=True);path.write_bytes(b'original-raw');before=inventory(source)
            output=root/'new';proof=clone_raw(source,output,before)
            self.assertTrue(proof['all_original_bytes_verified']);self.assertEqual(inventory(output),before)
            (output/'new-audit.json').write_text('{}');self.assertEqual(inventory(source),before)
            with self.assertRaises(ValueError):clone_raw(source,output,before)
    def test_already_audited_tree_is_not_rewritten(self):
        with tempfile.TemporaryDirectory()as d:
            with self.assertRaises(ValueError):clone_raw(Path(d),Path(d)/'new',[dict(path='control32/report.json')])
    def test_recovery_only_audits_readout_and_first_fits_bias(self):
        calls=[];receipts=execute_recovery(lambda *v:calls.append(('run',)+v),lambda *v:calls.append(('backup',)+v))
        self.assertEqual(calls[:3],[('run','readout','audit-saved'),('backup','readout','evaluate',0),('run','bias','fit')]);self.assertEqual(len(receipts),3)
        self.assertNotIn(('run','readout','fit'),calls);self.assertNotIn(('run','readout','evaluate'),calls)
    def test_recovery_clock_carries_all_actual_parent_usage(self):
        lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(g.AUTH),GPU_count=1,job_id='456',node='node',source='source',start_unix=0.,stop_unix=37800.,prior_award_GPU_seconds=5394,other_reserved_GPU_seconds=0,paired_authorization_sha256=g.sha(g.PAIR_AUTH),paired_recovery_sha256=g.sha(g.PAIR_RECOVERY_AUTH),paired_parent_job='8540802',prior_phase_GPU_seconds=5394)
        g.resource(lease)
        with self.assertRaises(ValueError):g.resource(dict(lease,stop_unix=37801.))
        with self.assertRaises(ValueError):g.resource(dict(lease,prior_phase_GPU_seconds=0))
