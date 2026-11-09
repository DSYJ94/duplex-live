import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.personaplex import matched_guard as g


class MatchedGuard(unittest.TestCase):
    def setUp(self):
        self.start=g.stamp('2026-11-09T12:00:00');self.end=self.start+86400
        self.lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,
            authorization_sha256=g.sha(g.AUTH),job_id='12345',node='nid1',source='a'*40,
            GPU_count=2,prior_GPU_seconds=327,maximum_seconds=86400,
            start_unix=self.start,stop_unix=self.end)
        self.job=dict(JobId='12345',Account=g.ACCOUNT,JobState='RUNNING',
            AllocTRES='cpu=64,gres/gpu=2',NumNodes='1',NodeList='nid1',Partition='workq',
            StartTime='2026-11-09T12:00:00',EndTime='2026-11-10T12:00:00')
        self.guard={k:self.lease[k] for k in ('job_id','account','node','source','authorization_sha256','stop_unix','GPU_count','prior_GPU_seconds')}
        self.guard.update(checked_unix=self.start+5,expires_unix=self.start+50)
        self.env=dict(SLURM_JOB_ID='12345',SLURM_JOB_ACCOUNT=g.ACCOUNT)

    def test_exact_new_scope_passes_but_original_one_gpu_scope_is_not_changed(self):
        g.validate_job(self.job,self.lease,self.start+10)
        g.ensure(self.lease,self.guard,self.start+10,self.env,needed=600)
        from training.venus import isambard_guard as old
        with self.assertRaises(ValueError):old.resource(self.lease)

    def test_wrong_actual_gpu_count_identity_terminal_state_and_clock_rejected(self):
        for field,value in [('JobId','12'),('Account','other'),('JobState','COMPLETED'),
            ('AllocTRES','gres/gpu=1'),('NumNodes','2'),('NodeList','nid2'),('EndTime','2026-11-10T12:00:01')]:
            changed=dict(self.job);changed[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):g.validate_job(changed,self.lease,self.start+10)

    def test_aggregate_budget_is_carried_and_cannot_be_reset_or_exceeded(self):
        for prior in (0,326,g.LIMIT-2*86400+1):
            changed=dict(self.lease,prior_GPU_seconds=prior)
            with self.subTest(prior=prior),self.assertRaises(ValueError):g.resource(changed)
        changed=dict(self.lease,prior_GPU_seconds=g.LIMIT-2*86400)
        g.resource(changed)

    def test_missing_expired_or_wrong_guard_never_admits_model(self):
        for changed,now,env,needed in [(dict(self.guard,source='b'*40),self.start+10,self.env,0),
            (self.guard,self.start+51,self.env,0),(self.guard,self.start+10,{},0),
            (self.guard,self.start+10,self.env,86400)]:
            with self.assertRaises(RuntimeError):g.ensure(self.lease,changed,now,env,needed)

    def test_observation_timeout_preserves_original_expiry_and_never_restarts(self):
        import subprocess
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'guard.json';path.write_text(json.dumps(self.guard));before=path.read_bytes()
            with patch.object(g.time,'time',return_value=self.start+20),patch.object(g.subprocess,'check_output',side_effect=subprocess.TimeoutExpired('scontrol',10)):
                self.assertEqual(g.refresh(self.lease,path,self.start+5),self.start+5)
            self.assertEqual(before,path.read_bytes())
            with patch.object(g.time,'time',return_value=self.start+50),patch.object(g.subprocess,'check_output',side_effect=subprocess.TimeoutExpired('scontrol',1)):
                with self.assertRaises(subprocess.TimeoutExpired):g.refresh(self.lease,path,self.start+5)

    def test_other_project_job_stops_without_refreshing_lease(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'guard.json';path.write_text('preserve')
            with patch.object(g.time,'time',return_value=self.start+10),patch.object(g.subprocess,'check_output',side_effect=['unused','12345\n99999\n']):
                with self.assertRaisesRegex(RuntimeError,'Another project job'):g.refresh(self.lease,path,self.start+5)
            self.assertEqual(path.read_text(),'preserve')


if __name__=='__main__':unittest.main()
