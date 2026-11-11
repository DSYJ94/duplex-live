import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from training.personaplex import matched_rpc_startup as s,matched_rpc_startup_submit as submit
from training.personaplex import matched_rpc_job as job,matched_guard as g,matched_resume as resume
from training.personaplex.matched_failed_state import authority as post
from training.personaplex.matched_normal88 import authority as normal
from training.personaplex.matched_storage_recovery import authority as storage
from evals.harness import test_personaplex_matched_guard as fixture


def phase():
    return dict(continuation_authorization_sha256=g.sha(g.EXTENSION),continuation_interval=9,
        postupdate_recovery_authorization_sha256=post(),normal88_authorization_sha256=normal(),
        storage_recovery_authorization_sha256=storage(),rpc_recovery_authorization_sha256=s.recovery.authority(),
        rpc_startup_authorization_sha256=s.authority())


class ApprovedStartup(unittest.TestCase):
    def test_mount_target_is_precreated_and_reusable_before_parallel_children(self):
        with tempfile.TemporaryDirectory() as root:
            target=job.prepare_mount_target(Path(root))
            self.assertTrue(target.is_dir())
            self.assertEqual(job.prepare_mount_target(Path(root)),target)
            self.assertEqual(list(target.iterdir()),[])

    def test_mount_target_rejects_files_and_aliases_without_removing_them(self):
        with tempfile.TemporaryDirectory() as root:
            target=Path(root)/'runs';target.write_text('preserve')
            with self.assertRaises(FileExistsError):job.prepare_mount_target(Path(root))
            self.assertEqual(target.read_text(),'preserve')
            target.unlink();target.symlink_to(Path(root),target_is_directory=True)
            with self.assertRaises(ValueError):job.prepare_mount_target(Path(root))
            self.assertTrue(target.is_symlink())

    def test_exact_grant_and_no_extra_interval_or_money(self):
        p=phase();self.assertEqual(g.phase_limit(p),112*3600)
        self.assertEqual(g.phase_limit(dict(p,continuation_interval=10)),112*3600)
        for number in (8,11):
            with self.assertRaises(ValueError):g.phase_limit(dict(p,continuation_interval=number))
        with self.assertRaises(ValueError):g.phase_limit(dict(p,rpc_startup_authorization_sha256='wrong'))
        p.pop('rpc_startup_authorization_sha256')
        with self.assertRaises(ValueError):g.phase_limit(dict(p,continuation_interval=10))

    def test_missing_scope_stops_before_preflight_or_scheduler(self):
        with patch.object(s,'authority',side_effect=ValueError('missing')),patch.object(submit.subprocess,'check_output') as query:
            with self.assertRaisesRegex(ValueError,'missing'):submit.submit('candidate')
            query.assert_not_called()
        args=SimpleNamespace(source='candidate',inputs_sha=job.DATA_SHA)
        with patch.object(job.subprocess,'check_output',return_value='candidate\n'),patch.object(s,'authority',side_effect=ValueError('missing')),patch.object(job,'preflight') as prepare:
            with self.assertRaisesRegex(ValueError,'missing'):job.main(args,startup_recovery=True)
            prepare.assert_not_called()

    def test_startup_grant_is_bound_into_live_guard(self):
        f=fixture.MatchedGuard();f.setUp();lease=dict(f.lease,**phase(),prior_GPU_seconds=135137)
        guard=dict(f.guard,**phase(),prior_GPU_seconds=135137)
        g.ensure(lease,guard,f.start+10,f.env,600)
        guard[s.KEY]='wrong'
        with self.assertRaisesRegex(RuntimeError,'identity'):g.ensure(lease,guard,f.start+10,f.env,600)
        guard[s.KEY]=s.authority()
        with self.assertRaises(RuntimeError):g.ensure(lease,guard,f.start+51,f.env,600)

    def test_actual_failed7_time_must_carry_before_positive8(self):
        rows=[]
        for ident,value in s.history(job.HISTORY).items():
            rows.append('|'.join([ident,'duplex-matched-test',g.ACCOUNT,value['state'],'0:0' if value['state']=='COMPLETED' else '1:0',str(value['elapsed']),'gres/gpu='+str(value['GPUs'])]))
        rows.append('90008|duplex-matched-eight|brics.u6ada|COMPLETED|0:0|7200|gres/gpu=2')
        ordered=[dict(lease=dict(phase(),job_id='90008',prior_GPU_seconds=135137,maximum_seconds=86400))]
        text='\n'.join(rows)
        self.assertEqual(resume.allocation_carry(text,ordered,'90009'),149537)
        for bad in (text.replace('|1:0|43|','|1:0|42|'),text.replace('|1:0|30|','|1:0|29|'),text.replace('COMPLETED|0:0|7200','FAILED|1:0|7200')):
            with self.assertRaises(ValueError):resume.allocation_carry(bad,ordered,'90009')

    def test_new_exclusive_submission_targets_eight_and_no_requeue(self):
        cmd=submit.command(Path('/allocated/work'),'candidate')
        self.assertIn('--no-requeue',cmd);self.assertIn('--gpus=2',cmd)
        self.assertIn('--job-name=duplex-matched-rpc-recovery9-v1',cmd)
        self.assertIn('matched_rpc_startup_job',cmd[-1])
        self.assertEqual(s.first_interval({s.KEY:s.authority()}),9)
        with self.assertRaises(ValueError):s.first_interval({s.KEY:'wrong'})

if __name__=='__main__':unittest.main()
