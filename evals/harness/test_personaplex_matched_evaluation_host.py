import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex import matched_evaluation_host as e,matched_guard as g
from training.personaplex.matched_failed_state import authority as post
from training.personaplex.matched_normal88 import authority as normal


class FinishedParentEvaluation(unittest.TestCase):
    def test_positive_complete_window_can_enter_evaluation_without_optimizer_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ex=root/'execution-123';ex.mkdir();grant=e.authority()
            (root/'phase-result.json').write_text(json.dumps(dict(source='worker',all_four_complete=True,partial_lineages=[])))
            (ex/'lease.json').write_text(json.dumps(dict(source='worker',job_id='123',continuation_interval=6,
                storage_recovery_authorization_sha256=grant)))
            (ex/'terminal.json').write_text(json.dumps(dict(source='worker',job_id='123',actual_exit_code=0)))
            (root/'storage-recovery-initial.json').write_text(json.dumps(dict(source='worker',authorization_sha256=grant,failed_job='8274714')))
            train=root/'personaplex/narrow/train';train.mkdir(parents=True);terminal=train/'training-terminal.json'
            terminal.write_text(json.dumps(dict(source='worker',actual_new_updates=84)))
            with patch.object(e.r,'beneath',return_value=root),patch.object(e,'completed_states') as complete:
                self.assertEqual(len(e.completed_parent(root,g.sha(root/'phase-result.json'))),1)
                complete.assert_called_once_with(root,'worker')
                terminal.write_text(json.dumps(dict(source='worker',actual_new_updates=0)))
                with self.assertRaisesRegex(ValueError,'positive updates'):e.completed_parent(root,g.sha(root/'phase-result.json'))

    def test_failed_or_partial_parent_cannot_launch_evaluation(self):
        for complete,code in ((False,0),(True,1)):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);ex=root/'execution-123';ex.mkdir()
                (root/'phase-result.json').write_text(json.dumps(dict(source='worker',all_four_complete=complete,partial_lineages=[])))
                (ex/'lease.json').write_text(json.dumps(dict(source='worker',job_id='123')))
                (ex/'terminal.json').write_text(json.dumps(dict(source='worker',job_id='123',actual_exit_code=code)))
                with patch.object(e.r,'beneath',return_value=root),self.assertRaisesRegex(ValueError,'successful full training'):
                    e.completed_parent(root,g.sha(root/'phase-result.json'))

    def test_new_allocation_uses_actual_new_clocks_and_all_carried_usage(self):
        parent=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(g.AUTH),
            continuation_authorization_sha256=g.sha(g.EXTENSION),continuation_interval=6,
            postupdate_recovery_authorization_sha256=post(),normal88_authorization_sha256=normal(),
            storage_recovery_authorization_sha256=e.authority(),job_id='123',source='native-worker',node='oldnode',
            start_unix=1,stop_unix=3601,maximum_seconds=3600,GPU_count=2,prior_GPU_seconds=44709)
        job=dict(JobId='456',Account=g.ACCOUNT,JobState='RUNNING',AllocTRES='gres/gpu=2',NumNodes='1',
            NodeList='newnode',Partition='workq',StartTime='2026-11-11T00:00:00',EndTime='2026-11-11T03:20:00')
        with patch.object(e.time,'time',return_value=g.stamp(job['StartTime'])+10):
            lease=e.new_allocation(parent,'evaluation-controller',job,50109)
        self.assertEqual(lease['prior_GPU_seconds'],50109);self.assertEqual(lease['continuation_interval'],7)
        self.assertEqual(lease['maximum_seconds'],12000);self.assertEqual(lease['node'],'newnode')
        self.assertEqual(parent['job_id'],'123');self.assertEqual(lease['source'],'native-worker')


if __name__=='__main__':unittest.main()
