import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from training.personaplex import matched_storage_recovery as s,matched_guard as g,matched_resume as r
from training.personaplex.matched_failed_state import authority as post
from training.personaplex.matched_normal88 import authority as normal
from training.personaplex.matched_storage_job import HISTORY


class ExactStorageRecovery(unittest.TestCase):
    def phase(self,interval):
        return dict(continuation_authorization_sha256=g.sha(g.EXTENSION),continuation_interval=interval,
            postupdate_recovery_authorization_sha256=post(),normal88_authorization_sha256=normal(),
            storage_recovery_authorization_sha256=s.authority())

    def test_actual_grant_allows_only_eight_intervals_with_same112_hours(self):
        for i in (6,7,8):self.assertEqual(g.phase_limit(self.phase(i)),112*3600)
        with self.assertRaises(ValueError):g.phase_limit(self.phase(9))
        bad=self.phase(6);bad['storage_recovery_authorization_sha256']='changed'
        with self.assertRaises(ValueError):g.phase_limit(bad)
        value=json.loads(s.AUTH.read_text());value['new_external_model_API_calls']=1
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'grant';p.write_text(json.dumps(value))
            with patch.object(s,'AUTH',p),self.assertRaises(ValueError):s.authority()

    def test_only_original_normal112_104_references_are_allowed(self):
        for base,(step,digest) in s.STATES.items():
            a=SimpleNamespace(base=base,capacity='narrow',checkpoint_sha=digest,
                checkpoint=str(s.FAILED_ROOT/base/'narrow/train/live.pt'))
            self.assertTrue(s.reference_allowed(a,s.PREDECESSOR))
            self.assertFalse(s.reference_allowed(a,'other-producer'))
            a.checkpoint=a.checkpoint.replace('live.pt','interrupted.pt')
            self.assertFalse(s.reference_allowed(a,s.PREDECESSOR))
            a.checkpoint=str(s.FAILED_ROOT/base/'narrow/train/live.pt');a.capacity='broad'
            self.assertFalse(s.reference_allowed(a,s.PREDECESSOR))

    def test_future_positive_resume_counts_all_five_failures_and_capture(self):
        lines=[]
        for job,v in HISTORY.items():
            lines.append('|'.join([job,'duplex-matched-fixture',g.ACCOUNT,v['state'],
                '0:0' if job=='8216374' else '1:0',str(v['elapsed']),'gres/gpu='+str(v['GPUs'])]))
        lines.append('99990|duplex-matched-storage|brics.u6ada|COMPLETED|0:0|7200|gres/gpu=2')
        ordered=[dict(lease=dict(job_id='99990',prior_GPU_seconds=44709,maximum_seconds=86400,
            storage_recovery_authorization_sha256=s.authority()))]
        text='\n'.join(lines)
        self.assertEqual(r.allocation_carry(text,ordered,'99991'),59109)
        with self.assertRaises(ValueError):r.allocation_carry(text.replace('|5325|','|5324|'),ordered,'99991')
        with self.assertRaises(ValueError):r.allocation_carry(text.replace('COMPLETED|0:0|7200','FAILED|1:0|7200'),ordered,'99991')


if __name__=='__main__':unittest.main()
