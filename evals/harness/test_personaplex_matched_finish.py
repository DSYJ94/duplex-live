import unittest
from training.personaplex import matched_finish as f,matched_guard as g
from training.personaplex.matched_normal88 import authority as normal_authority
from training.personaplex.matched_failed_state import authority as post_authority


class SameAllocationEvaluation(unittest.TestCase):
    def lease(self):
        return dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(g.AUTH),
            continuation_authorization_sha256=g.sha(g.EXTENSION),continuation_interval=6,
            postupdate_recovery_authorization_sha256=post_authority(),normal88_authorization_sha256=normal_authority(),
            job_id='99999',source='immutable-trainer',controller_source='controller',node='fixture-node',
            start_unix=100000,stop_unix=186400,maximum_seconds=86400,GPU_count=2,prior_GPU_seconds=200000)

    def test_new_evaluation_source_preserves_actual_allocation_clocks_and_prior_cost(self):
        before=self.lease();after=f.evaluation_lease(before,'evaluation-producer','99999')
        self.assertEqual(after['trainer_source'],'immutable-trainer');self.assertEqual(after['source'],'evaluation-producer')
        for key in ('start_unix','stop_unix','maximum_seconds','GPU_count','prior_GPU_seconds','continuation_interval'):
            self.assertEqual(after[key],before[key])
        self.assertEqual(before['source'],'immutable-trainer')

    def test_wrong_interval_job_or_over_budget_lease_cannot_start_evaluation(self):
        before=self.lease()
        for value in (dict(before,continuation_interval=5),dict(before,job_id='other'),dict(before,prior_GPU_seconds=300000)):
            with self.assertRaises(ValueError):f.evaluation_lease(value,'evaluation-producer','99999')


if __name__=='__main__':unittest.main()
