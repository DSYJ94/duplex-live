import json,subprocess,unittest
from types import SimpleNamespace
from training.personaplex.matched_observer import probe


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


if __name__=='__main__':unittest.main()
