import io,json,subprocess,unittest
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace
from training.personaplex.matched_observer import probe
from training.personaplex import matched_observer as observer


class BoundedMonitoring(unittest.TestCase):
    root='/home/u6ada/dsyj94.u6ada/duplex-native-recovery-20261109-v1/runs/fixture-999'

    def test_fresh_transport_and_actual_timeout_not_unchanged_live_claim(self):
        def runner(cmd,**kwargs):
            self.assertEqual(cmd[cmd.index('-S')+1],'none');self.assertIn('25s',cmd);self.assertEqual(kwargs['timeout'],35)
            return SimpleNamespace(returncode=0,stdout=json.dumps(dict(job='999',host_terminal=dict(actual_exit_code=1))),stderr='')
        self.assertEqual(probe('999',self.root,runner)['host_terminal']['actual_exit_code'],1)
        def hung(*args,**kwargs):raise subprocess.TimeoutExpired('fixture-ssh',35)
        with self.assertRaises(subprocess.TimeoutExpired):probe('999',self.root,hung)

    def test_failed_access_or_wrong_job_is_not_a_healthy_observation(self):
        for result in (SimpleNamespace(returncode=255,stdout='',stderr='expired auth'),
                       SimpleNamespace(returncode=0,stdout='{"job":"other"}',stderr='')):
            with self.assertRaises((RuntimeError,ValueError)):probe('999',self.root,lambda *a,**k:result)

    def test_one_failed_read_is_reported_before_bounded_read_only_recovery(self):
        a=SimpleNamespace(job='999',root=self.root,once=False);out=io.StringIO()
        fresh=dict(job='999',actual_UTC='2026-11-10T22:50:00Z',host_terminal=dict(actual_exit_code=0),queue=dict(actual_exit=0,stdout=''))
        with patch.object(observer,'probe',side_effect=[subprocess.TimeoutExpired('ssh',35),fresh]) as read,patch.object(observer.time,'sleep'),redirect_stdout(out):
            observer.main(a)
        events=[json.loads(s) for s in out.getvalue().splitlines()]
        self.assertEqual(read.call_count,2);self.assertEqual(events[0]['event'],'MONITORING_BLOCKED')
        self.assertFalse(events[0]['job_restart_requested']);self.assertEqual(events[1]['event'],'MONITORING_RESTORED')

    def test_two_failed_reads_stop_instead_of_silent_infinite_retries(self):
        a=SimpleNamespace(job='999',root=self.root,once=False)
        with patch.object(observer,'probe',side_effect=subprocess.TimeoutExpired('ssh',35)) as read,patch.object(observer.time,'sleep'),redirect_stdout(io.StringIO()),self.assertRaises(subprocess.TimeoutExpired):
            observer.main(a)
        self.assertEqual(read.call_count,2)


if __name__=='__main__':unittest.main()
