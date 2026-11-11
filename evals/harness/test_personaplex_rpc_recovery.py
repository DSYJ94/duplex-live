import unittest
from types import SimpleNamespace
from unittest.mock import patch
from training.personaplex import matched_rpc_recovery as r
from training.personaplex import matched_rpc_job as job


class ExactFailureStates(unittest.TestCase):
    def test_missing_amendment_stops_before_storage_or_model_preparation(self):
        args=SimpleNamespace(source='candidate',inputs_sha=job.DATA_SHA)
        with patch.object(job.subprocess,'check_output',return_value='candidate\n'),\
             patch.object(r,'authority',side_effect=ValueError('pending amendment')),\
             patch.object(job,'storage_check') as storage,patch.object(job,'preflight') as models:
            with self.assertRaisesRegex(ValueError,'pending amendment'):job.main(args)
            storage.assert_not_called();models.assert_not_called()

    def test_exact_selected_reference_requires_actual_owner_grant(self):
        for key in r.STATES:
            path,step,digest=r.selected(*key)
            args=SimpleNamespace(base=key[0],capacity=key[1],checkpoint=str(path),checkpoint_sha=digest)
            with patch.object(r,'authority',side_effect=ValueError('pending')):
                with self.assertRaisesRegex(ValueError,'pending'):r.reference_allowed(args,r.PREDECESSOR)
            with patch.object(r,'authority',return_value='actual-grant'):
                self.assertTrue(r.reference_allowed(args,r.PREDECESSOR))
                args.checkpoint_sha='0'*64;self.assertFalse(r.reference_allowed(args,r.PREDECESSOR))

    def test_normal56_rollback_and_unrelated_sources_are_not_allowed(self):
        path,step,digest=r.selected('personaplex','broad')
        args=SimpleNamespace(base='personaplex',capacity='broad',checkpoint=str(path.with_name('live.pt')),checkpoint_sha=digest)
        with patch.object(r,'authority') as grant:
            self.assertFalse(r.reference_allowed(args,r.PREDECESSOR))
            args.checkpoint=str(path);self.assertFalse(r.reference_allowed(args,'another-source'))
            grant.assert_not_called()

    def test_steps_pin_both_completed_narrows_and_latest_broads(self):
        self.assertEqual({k:v[0] for k,v in r.STATES.items()},
            {('personaplex','narrow'):196,('context-spanning','narrow'):196,
             ('personaplex','broad'):62,('context-spanning','broad'):63})


if __name__=='__main__':unittest.main()
