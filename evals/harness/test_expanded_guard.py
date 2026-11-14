import unittest
from training.personaplex.expanded_guard import AUTH,KEY,ACCOUNT,sha,ensure,resource

class ExpandedGuardTest(unittest.TestCase):
    def fixture(self):
        lease=dict(provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),GPU_count=4,job_id='123',node='node1',source='source1',start_unix=0.,stop_unix=1000.,prior_award_GPU_seconds=100.,other_reserved_GPU_seconds=20.)
        guard={k:lease[k]for k in ('job_id','account','node','source','authorization_sha256','GPU_count','start_unix','stop_unix')};guard.update(provider_job_verified=True,terminal=False,heartbeat_unix=100.,provider_checked_unix=1.)
        return lease,guard,dict(SLURM_JOB_ID='123',SLURM_JOB_ACCOUNT=ACCOUNT)
    def test_unknown_observation_does_not_extend_deadline_or_interrupt_live_worker(self):
        lease,guard,env=self.fixture();guard['observation_error']='TimeoutExpired'
        ensure(lease,guard,now=110.,environ=env)
        with self.assertRaises(RuntimeError):ensure(lease,guard,now=821.,environ=env)
    def test_wrong_allocation_expired_heartbeat_and_terminal_fail(self):
        lease,guard,env=self.fixture()
        for changed in [dict(guard,job_id='456'),dict(guard,heartbeat_unix=40.),dict(guard,terminal=True),dict(guard,provider_job_verified=False)]:
            with self.assertRaises(RuntimeError):ensure(lease,changed,now=100.,environ=env)
        with self.assertRaises(ValueError):resource(dict(lease,GPU_count=8))
        with self.assertRaises(ValueError):resource(dict(lease,prior_award_GPU_seconds=20000*3600))
