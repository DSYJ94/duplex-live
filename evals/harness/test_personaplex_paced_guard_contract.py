import unittest
from training.personaplex import paced_guard_contract as c


class SeparateClocks(unittest.TestCase):
    def setUp(self):
        self.lease=dict(job_id='fixture',account='brics.u6ada',node='fixture-node',GPU_count=2,
            stop_unix=90000,maximum_seconds=86400,prior_GPU_seconds=135137)
        self.provider=dict(self.lease,observed_unix=1000,state='RUNNING',only_project_job=True)
        self.local=dict(self.lease,observed_unix=1060,process_alive=True,same_pid_start=True,
            same_job_cgroup=True,same_UID=True,storage_admission_valid=True)

    def test_no_fast_retry_even_after_a_failed_provider_attempt(self):
        self.assertTrue(c.provider_due(1000,None))
        self.assertFalse(c.provider_due(1059.99,1000))
        self.assertTrue(c.provider_due(1060,1000))

    def test_provider_stamp_is_not_relabelled_by_local_renewal(self):
        record=c.validate(self.lease,self.provider,self.local,1060)
        self.assertEqual(record['provider_observed_unix'],1000)
        self.assertEqual(record['local_observed_unix'],1060)
        self.assertFalse(record['runtime_integration'])

    def test_expired_either_clock_stops_despite_other_fresh_evidence(self):
        for provider,local in ((dict(self.provider,observed_unix=969),self.local),
                               (self.provider,dict(self.local,observed_unix=1014))):
            with self.assertRaises(RuntimeError):c.validate(self.lease,provider,local,1060)

    def test_cancellation_overlap_deadline_or_local_identity_change_stops(self):
        for key,value in [('state','COMPLETED'),('only_project_job',False),('stop_unix',90001)]:
            with self.assertRaises(RuntimeError):c.validate(self.lease,dict(self.provider,**{key:value}),self.local,1060)
        for key in ('process_alive','same_pid_start','same_job_cgroup','same_UID','storage_admission_valid'):
            with self.assertRaises(RuntimeError):c.validate(self.lease,self.provider,dict(self.local,**{key:False}),1060)

    def test_original_time_and_gpu_budget_cannot_reset(self):
        with self.assertRaises(RuntimeError):c.validate(dict(self.lease,prior_GPU_seconds=400000),self.provider,self.local,1060)
        with self.assertRaises(RuntimeError):c.validate(self.lease,self.provider,self.local,90000-180)

    def test_invalid_clocks_cannot_bypass_budget_or_pacing(self):
        for value in (float('nan'),float('inf'),-1,True):
            for key in ('stop_unix','maximum_seconds','prior_GPU_seconds'):
                with self.assertRaises(RuntimeError):
                    c.validate(dict(self.lease,**{key:value}),self.provider,self.local,1060)
        for value in (float('nan'),float('inf'),1061,True):
            with self.assertRaises(ValueError):c.provider_due(1060,value)

    def test_missing_identity_is_not_equal_valid_identity(self):
        for key in ('job_id','account','node'):
            lease=dict(self.lease);provider=dict(self.provider);local=dict(self.local)
            for record in (lease,provider,local):record.pop(key)
            with self.assertRaises(RuntimeError):c.validate(lease,provider,local,1060)

if __name__=='__main__':unittest.main()
