import unittest
from training.personaplex.expanded_guard import AUTH,REFERENCE_AUTH,OFFICIAL_AUTH,FEATURE_AUTH,RESTORE_AUTH,TRAIN_FAILURE_AUTH,CORPUS_REVIEW_AUTH,CORPUS_REPAIR_AUTH,KEY,ACCOUNT,sha,ensure,resource

class ExpandedGuardTest(unittest.TestCase):
    def fixture(self):
        lease=dict(provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),GPU_count=4,job_id='123',node='node1',source='source1',start_unix=0.,stop_unix=1000.,prior_award_GPU_seconds=100.,other_reserved_GPU_seconds=20.)
        guard={k:lease[k]for k in ('job_id','account','node','source','authorization_sha256','GPU_count','start_unix','stop_unix')};guard.update(provider_job_verified=True,terminal=False,heartbeat_unix=100.,provider_checked_unix=1.)
        return lease,guard,dict(SLURM_JOB_ID='123',SLURM_JOB_ACCOUNT=ACCOUNT)
    def test_unknown_observation_does_not_extend_deadline_or_interrupt_live_worker(self):
        lease,guard,env=self.fixture();guard['observation_error']='TimeoutExpired'
        ensure(lease,guard,now=110.,environ=env)
        with self.assertRaises(RuntimeError):ensure(lease,guard,now=821.,environ=env)
    def test_fresh_guard_requires_continuation_before_deadline(self):
        lease,guard,env=self.fixture();guard['heartbeat_unix']=800.
        with self.assertRaises(TimeoutError):ensure(lease,guard,needed=30,now=800.,environ=env)

    def test_wrong_allocation_expired_heartbeat_and_terminal_fail(self):
        lease,guard,env=self.fixture()
        for changed in [dict(guard,job_id='456'),dict(guard,heartbeat_unix=40.),dict(guard,terminal=True),dict(guard,provider_job_verified=False)]:
            with self.assertRaises(RuntimeError):ensure(lease,changed,now=100.,environ=env)
        with self.assertRaises(ValueError):resource(dict(lease,GPU_count=8))
        with self.assertRaises(ValueError):resource(dict(lease,prior_award_GPU_seconds=20000*3600))

    def test_single_gpu_requires_its_exact_separate_reference_scope(self):
        lease,_,_=self.fixture();single=dict(lease,GPU_count=1)
        with self.assertRaises(ValueError):resource(single)
        reference=dict(single,reference_authorization_sha256=sha(REFERENCE_AUTH))
        resource(reference)
        with self.assertRaises(ValueError):resource(dict(reference,GPU_count=4))
        with self.assertRaises(ValueError):resource(dict(reference,stop_unix=7201))
        with self.assertRaises(ValueError):resource(dict(reference,reference_authorization_sha256='changed'))
        official=dict(single,official_authorization_sha256=sha(OFFICIAL_AUTH),stop_unix=18000.)
        resource(official)
        with self.assertRaises(ValueError):resource(dict(official,reference_authorization_sha256=sha(REFERENCE_AUTH)))
        with self.assertRaises(ValueError):resource(dict(official,stop_unix=18001.))

    def test_feature_restore_scope_stays_zero_update_and_single_gpu(self):
        lease,_,_=self.fixture();feature=dict(lease,GPU_count=1,feature_authorization_sha256=sha(FEATURE_AUTH),stop_unix=21600.)
        resource(feature)
        with self.assertRaises(ValueError):resource(dict(feature,GPU_count=4))
        with self.assertRaises(ValueError):resource(dict(feature,stop_unix=21601.))
        with self.assertRaises(ValueError):resource(dict(feature,official_authorization_sha256=sha(OFFICIAL_AUTH)))

    def test_restore_only_recovery_keeps_remaining_clock(self):
        lease,_,_=self.fixture();restore=dict(lease,GPU_count=1,restore_authorization_sha256=sha(RESTORE_AUTH),stop_unix=16200.)
        resource(restore)
        with self.assertRaises(ValueError):resource(dict(restore,stop_unix=16201.))
        with self.assertRaises(ValueError):resource(dict(restore,feature_authorization_sha256=sha(FEATURE_AUTH)))

    def test_train_failure_scope_is_separate_complete_four_gpu_and_bounded(self):
        lease,_,_=self.fixture()
        collect=dict(lease,train_failure_authorization_sha256=sha(TRAIN_FAILURE_AUTH),stop_unix=43200.)
        resource(collect)
        for change in (dict(GPU_count=1),dict(stop_unix=43201.),
                       dict(train_failure_authorization_sha256='changed'),
                       dict(reference_authorization_sha256=sha(REFERENCE_AUTH))):
            with self.assertRaises(ValueError):resource(dict(collect,**change))

    def test_corpus_review_scope_cannot_mix_or_extend_existing_collection(self):
        lease,_,_=self.fixture()
        review=dict(lease,corpus_review_authorization_sha256=sha(CORPUS_REVIEW_AUTH),
                    corpus_review_parent_job='8553179',stop_unix=43200.)
        resource(review)
        for change in (dict(GPU_count=1),dict(stop_unix=43201.),dict(corpus_review_parent_job='wrong'),
                       dict(corpus_review_authorization_sha256='changed'),
                       dict(train_failure_authorization_sha256=sha(TRAIN_FAILURE_AUTH)),
                       dict(reference_authorization_sha256=sha(REFERENCE_AUTH))):
            with self.assertRaises(ValueError):resource(dict(review,**change))

    def test_format_repair_only_uses_remaining_combined_time(self):
        lease,_,_=self.fixture()
        review=dict(lease,corpus_review_authorization_sha256=sha(CORPUS_REVIEW_AUTH),
                    corpus_review_parent_job='8553179',corpus_review_repair_sha256=sha(CORPUS_REPAIR_AUTH),stop_unix=42600.)
        resource(review)
        with self.assertRaises(ValueError):resource(dict(review,stop_unix=42601.))
        with self.assertRaises(ValueError):resource(dict(review,corpus_review_repair_sha256='wrong'))
