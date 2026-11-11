"""Exercise real heartbeat refresh with scaled scheduler RPC latency."""
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex import matched_guard as g
from evals.harness import test_personaplex_matched_guard as fixtures


class SchedulerLatency(unittest.TestCase):
    def test_two_valid_slow_reads_fit_one_shared_budget_when_overlapped(self):
        fixture=fixtures.MatchedGuard();fixture.setUp()
        real_clock=time.monotonic;began=real_clock()
        def now():return fixture.start+10+(real_clock()-began)*100
        text=' '.join(k+'='+v for k,v in fixture.job.items())
        def rpc(command,**kwargs):
            delay=.12;allowed=kwargs['timeout']/100
            time.sleep(min(delay,allowed))
            if allowed<delay:raise subprocess.TimeoutExpired(command,kwargs['timeout'])
            return text if command[0]=='scontrol' else '12345\n'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'guard.json'
            with patch.object(g.time,'time',side_effect=now),patch.object(g.subprocess,'check_output',side_effect=rpc):
                checked=g.refresh(fixture.lease,path)
            value=json.loads(path.read_text())
            self.assertEqual(value['checked_unix'],checked)
            self.assertEqual(value['expires_unix'],checked+45)
            self.assertLess(checked,fixture.start+30)

    def test_partial_pair_never_refreshes_and_expired_window_still_raises(self):
        fixture=fixtures.MatchedGuard();fixture.setUp()
        def rpc(command,**kwargs):
            if command[0]=='squeue':raise subprocess.TimeoutExpired(command,1)
            return ' '.join(k+'='+v for k,v in fixture.job.items())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'guard.json';path.write_text('original')
            with patch.object(g.time,'time',return_value=fixture.start+20),patch.object(g.subprocess,'check_output',side_effect=rpc):
                self.assertEqual(g.refresh(fixture.lease,path,fixture.start+5),fixture.start+5)
            self.assertEqual(path.read_text(),'original')
            with patch.object(g.time,'time',return_value=fixture.start+50),patch.object(g.subprocess,'check_output',side_effect=rpc):
                with self.assertRaises(subprocess.TimeoutExpired):g.refresh(fixture.lease,path,fixture.start+5)
            self.assertEqual(path.read_text(),'original')

    def test_complete_pair_with_wrong_native_allocation_does_not_renew(self):
        fixture=fixtures.MatchedGuard();fixture.setUp()
        changed=dict(fixture.job,AllocTRES='gres/gpu=1')
        def rpc(command,**kwargs):
            return ' '.join(k+'='+v for k,v in changed.items()) if command[0]=='scontrol' else '12345\n'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'guard.json';path.write_text('original')
            with patch.object(g.time,'time',return_value=fixture.start+20),patch.object(g.subprocess,'check_output',side_effect=rpc):
                with self.assertRaises(ValueError):g.refresh(fixture.lease,path)
            self.assertEqual(path.read_text(),'original')


if __name__=='__main__':unittest.main()
