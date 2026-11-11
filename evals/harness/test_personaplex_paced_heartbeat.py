import subprocess
import unittest
from training.personaplex.paced_heartbeat import Heartbeat


class HeartbeatIntegration(unittest.TestCase):
    def setUp(self):
        self.now = 1000.
        self.mono = 0.
        self.calls = []
        self.failure = None
        self.lease = dict(job_id='123', account='brics.u6ada', node='fixture',
                          GPU_count=2, stop_unix=90000, maximum_seconds=86400,
                          prior_GPU_seconds=135137)
        self.provider = dict(self.lease, observed_unix=self.now, state='RUNNING', only_project_job=True)
        self.local = dict(self.lease, process_alive=True, same_pid_start=True,
                         same_job_cgroup=True, same_UID=True, storage_admission_valid=True)
        def remote():
            self.calls.append(self.mono)
            if self.failure: raise self.failure
            return dict(self.provider, observed_unix=self.now)
        self.engine = Heartbeat(self.lease, remote,
            lambda: dict(self.local, observed_unix=self.now),
            clock=lambda: self.now, monotonic=lambda: self.mono)

    def advance(self, seconds):
        self.now += seconds
        self.mono += seconds

    def test_local_checks_never_relabel_provider_time_or_poll_fast(self):
        self.engine.tick()
        for _ in range(5):
            self.advance(10)
            proof = self.engine.tick()
            self.assertEqual(proof['provider_observed_unix'], 1000)
            self.assertEqual(proof['local_observed_unix'], self.now)
        self.assertEqual(self.calls, [0])
        self.advance(10)
        self.assertEqual(self.engine.tick()['provider_observed_unix'], 1060)
        self.assertEqual(self.calls, [0, 60])

    def test_timeout_uses_only_original_unexpired_evidence_without_fast_retry(self):
        self.engine.tick()
        self.advance(60)
        self.failure = subprocess.TimeoutExpired('fixture', 20)
        proof = self.engine.tick()
        self.assertTrue(proof['provider_query_timed_out'])
        self.assertEqual(proof['expires_unix'], 1090)
        self.advance(20)
        self.engine.tick()
        self.assertEqual(self.calls, [0, 60])
        self.advance(11)
        with self.assertRaises(RuntimeError): self.engine.tick()
        self.failure = None
        self.advance(60)
        with self.assertRaises(RuntimeError): self.engine.tick()
        self.assertEqual(self.calls, [0, 60])

    def test_initial_timeout_never_creates_a_guard(self):
        self.failure = subprocess.TimeoutExpired('fixture', 20)
        with self.assertRaises(subprocess.TimeoutExpired): self.engine.tick()
        self.assertIsNone(self.engine.provider)
        self.assertTrue(self.engine.failed)

    def test_explicit_cancellation_cannot_fall_back_to_previous_success(self):
        self.engine.tick()
        self.advance(60)
        self.provider['state'] = 'CANCELLED'
        with self.assertRaises(RuntimeError): self.engine.tick()
        self.assertTrue(self.engine.failed)

    def test_local_process_failure_stops_before_next_provider_poll(self):
        self.engine.tick()
        self.advance(10)
        self.local['same_pid_start'] = False
        with self.assertRaises(RuntimeError): self.engine.tick()
        self.assertEqual(self.calls, [0])

    def test_monotonic_reversal_fails_instead_of_restarting_spacing(self):
        self.engine.tick()
        self.mono = -1
        with self.assertRaises(RuntimeError): self.engine.tick()

    def test_query_duration_counts_toward_expiry(self):
        def slow_remote():
            result = dict(self.provider)
            self.advance(91)
            return result
        self.engine.read_provider = slow_remote
        with self.assertRaises(RuntimeError): self.engine.tick()


if __name__ == '__main__': unittest.main()
