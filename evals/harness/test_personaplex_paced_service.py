import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from training.personaplex import paced_health_service as s, paced_finish as f
from training.personaplex.paced_heartbeat import Heartbeat


class ServiceWiring(unittest.TestCase):
    def test_unapproved_service_cannot_create_files_or_processes(self):
        with patch.object(s, 'authority', side_effect=FileNotFoundError('grant absent')), \
             patch.object(s.subprocess, 'Popen') as spawn, patch.object(s.Path, 'mkdir') as mkdir:
            with self.assertRaises(FileNotFoundError): s.start({}, '/fixture')
            spawn.assert_not_called()
            mkdir.assert_not_called()

    def test_guard_keeps_provider_time_separate_and_caps_expiry(self):
        lease = dict(job_id='123', account='brics.u6ada', node='fixture', GPU_count=2,
            stop_unix=90000, maximum_seconds=86400, prior_GPU_seconds=135137,
            controller_source='fixture-controller')
        parent = dict(cgroup='fixture-cgroup', uids=[1001]*4)
        provider = dict(lease, observed_unix=1000, state='RUNNING', only_project_job=True,
                        original_provider_receipt=dict(lease, checked_unix=1000, expires_unix=1045))
        local = dict(lease, observed_unix=1060, process_alive=True, same_pid_start=True,
            same_job_cgroup=True, same_UID=True, storage_admission_valid=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'parent-identity.json').write_text(json.dumps(parent))
            with patch.object(s, 'authority', return_value='fixture-grant'), \
                 patch.object(s, 'check_context'), patch.object(s.process, 'capture', return_value=parent), \
                 patch.object(s.observations, 'provider', return_value=provider), \
                 patch.object(s.observations, 'local', return_value=local), \
                 patch.object(s, 'Heartbeat', side_effect=lambda l,p,q,**kw: Heartbeat(l,p,q,clock=lambda:1060,monotonic=lambda:60,**kw)), \
                 patch.object(s.g, 'ensure') as ensure, \
                 patch.object(s.time, 'sleep', side_effect=InterruptedError('end fixture')):
                with self.assertRaises(InterruptedError): s.watch(lease, root)
            row = json.loads((root/'guard.json').read_text())
            self.assertEqual(row['checked_unix'], 1060)
            self.assertEqual(row['provider_observed_unix'], 1000)
            self.assertEqual(row['expires_unix'], 1090)
            self.assertEqual(row['service_source'], 'fixture-controller')
            ensure.assert_called_once_with(lease, row)

    def test_failed_native_controller_restores_dependency_without_restarting(self):
        original_resume = f.resume.matched_health
        original_finish = f.finish.matched_health
        def native(args):
            self.assertIs(f.resume.matched_health, s)
            self.assertIs(f.finish.matched_health, s)
            raise RuntimeError('actual child failed')
        with patch.object(s, 'authority', return_value='fixture-grant'), \
             patch.object(f.finish, 'main', side_effect=native) as run:
            with self.assertRaises(RuntimeError): f.main(SimpleNamespace(mode='train'))
            run.assert_called_once()
        self.assertIs(f.resume.matched_health, original_resume)
        self.assertIs(f.finish.matched_health, original_finish)

    def test_evaluation_mode_never_calls_training_controller(self):
        with patch.object(s, 'authority', return_value='fixture-grant'), \
             patch.object(f.evaluation, 'main') as evaluate, patch.object(f.finish, 'main') as train:
            args = SimpleNamespace(mode='evaluate')
            f.main(args)
            evaluate.assert_called_once_with(args)
            train.assert_not_called()


if __name__ == '__main__': unittest.main()
