import json,os,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex.expanded_guard import AUTH,KEY,ACCOUNT,sha
from training.personaplex.inference_guard import FrameGuard

class FrameChecks(unittest.TestCase):
    def test_all_frame_checks_stay_fresh_without_git_process_per_frame(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);head=root/'HEAD';head.write_text('a'*40+'\n');now=time.time()
            lease=dict(provider='Slurm',resource_key=KEY,account=ACCOUNT,authorization_sha256=sha(AUTH),GPU_count=4,job_id='123',node='node',source='a'*40,start_unix=now-10,stop_unix=now+600,prior_award_GPU_seconds=0,other_reserved_GPU_seconds=0)
            p=root/'lease.json';p.write_text(json.dumps(lease));g=root/'guard.json';snapshot=dict(lease,heartbeat_unix=now,provider_job_verified=True,terminal=False);g.write_text(json.dumps(snapshot))
            with patch.dict(os.environ,SLURM_JOB_ID='123',SLURM_JOB_ACCOUNT=ACCOUNT,DUPLEX_LIVE_GUARD_FILE=str(g)),patch('subprocess.check_output',side_effect=[str(head),''])as git:
                check=FrameGuard(p)
                for _ in range(4):check()
                self.assertEqual(git.call_count,2)
                g.write_text(json.dumps(dict(snapshot,heartbeat_unix=now-46)))
                with self.assertRaises(RuntimeError):check()
                g.write_text(json.dumps(snapshot));head.write_text('b'*40)
                with self.assertRaises(RuntimeError):check()
                head.write_text('a'*40);p.write_text(json.dumps(dict(lease,stop_unix=now+900)))
                with self.assertRaises(RuntimeError):check()
