import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from portable.context.native_saved_stage_controller import (
    AUTH_SHA, child_arguments, fresh_guard, monitor, terminate_child)


class ControllerTests(unittest.TestCase):
    def guard(self):
        now = time.time()
        return dict(execution_authorized=True, live_provider_deadline_verified=True,
            live_external_cost_guard_verified=True, single_gpu_across_providers_verified=True,
            checked_unix=now, deadline_unix=now+3600, provider='gcp',
            remaining_budget_usd=80., planned_maximum_cost_usd=6.,
            resource_id='123', authorization_sha256=AUTH_SHA)

    def test_fresh_guard_binds_authority_resource_and_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'guard.json'; value=self.guard()
            path.write_text(json.dumps(value));expected=fresh_guard(path)
            for changes in (dict(resource_id='other'),dict(provider='runpod'),
                            dict(authorization_sha256='a'*64),
                            dict(deadline_unix=value['deadline_unix']+1)):
                path.write_text(json.dumps(dict(value,**changes)))
                with self.subTest(changes=changes),self.assertRaises(ValueError):
                    fresh_guard(path,expected)

    def test_stale_guard_prevents_child_start_or_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'guard.json';value=self.guard()
            value['checked_unix']-=31;path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):fresh_guard(path)

    def test_monitor_terminates_only_its_child_on_guard_loss(self):
        child=SimpleNamespace(poll=lambda:None,returncode=None)
        with patch('portable.context.native_saved_stage_controller.fresh_guard',
                   side_effect=ValueError('stale')),patch(
                   'portable.context.native_saved_stage_controller.terminate_child') as stop:
            with self.assertRaises(ValueError):monitor(child,Path('unused'),self.guard())
            stop.assert_called_once_with(child)

    def test_nonzero_child_is_never_reported_as_complete(self):
        child=SimpleNamespace(poll=lambda:1,returncode=1)
        with patch('portable.context.native_saved_stage_controller.fresh_guard',return_value=self.guard()),patch(
                'portable.context.native_saved_stage_controller.terminate_child'):
            with self.assertRaises(RuntimeError):monitor(child,Path('unused'),self.guard())

    def test_child_termination_reaps_actual_owned_process(self):
        child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            terminate_child(child)
            self.assertIsNotNone(child.poll())
            self.assertNotEqual(child.returncode,0)
        finally:
            if child.poll() is None:child.kill();child.wait()

    def test_independent_process_command_preserves_exact_inputs_and_reference(self):
        args=SimpleNamespace(source=Path('/original'),snapshot=Path('/snapshot'),
            observation=Path('/observation'),guard=Path('/guard'),producer_commit='a'*40)
        first=child_arguments(args,'finalization',Path('/out1'),Path('/lease'))
        second=child_arguments(args,'independent_restore',Path('/out2'),Path('/lease'),Path('/out1/saved-stage-finalization.json'))
        self.assertEqual(first[0],sys.executable)
        self.assertNotIn('--reference',first)
        self.assertIn('--reference',second)
        self.assertEqual(second[-1],'/out1/saved-stage-finalization.json')
        for command in (first,second):
            self.assertNotIn('--actual-temporal-fit',command)
            self.assertNotIn('--resume-training',command)
            self.assertNotIn('--reload',command)


if __name__ == '__main__':
    unittest.main()
