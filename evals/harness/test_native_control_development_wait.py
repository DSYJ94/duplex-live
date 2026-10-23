import unittest
from datetime import datetime,timezone
from portable.context.native_control_development_wait import STAGES,ObservationUnavailable,ready,wait_ready,launch_payload,native_command


class DevelopmentWait(unittest.TestCase):
    def setUp(self):
        self.plan=dict(upstream_pid=4111,upstream_birth=123,pod_id='owned',source_commit='a'*40,
            latest_launch_utc=datetime.fromtimestamp(200,timezone.utc).isoformat())
        self.controller=dict(pid=4111,pod_id='owned',source_commit='a'*40,phase='complete40_precise_numerical',passed=False,stages=[])

    def completed(self):
        return dict(self.controller,phase='complete_restored_pilot_only',passed=True,
                    stages=[dict(name=n,exit_code=0) for n in STAGES])

    def test_completion_requires_every_stage_and_actual_controller_exit(self):
        c=self.completed()
        self.assertFalse(ready(dict(controller=c,birth=123),self.plan))
        self.assertTrue(ready(dict(controller=c,birth=None),self.plan))
        for bad in [dict(c,stages=c['stages'][:-1]),dict(c,pod_id='unrelated'),dict(c,source_commit='b'*40)]:
            with self.assertRaises(ValueError):ready(dict(controller=bad,birth=None),self.plan)
        with self.assertRaises(ValueError):ready(dict(controller=c,birth=124),self.plan)

    def test_read_timeout_is_retried_and_never_treated_as_terminal_or_launch(self):
        observations=[ObservationUnavailable('Network is unreachable'),TimeoutError(),
                      dict(controller=self.controller,birth=123),dict(controller=self.completed(),birth=None)]
        records=[];clock=[0]
        def observe():
            value=observations.pop(0)
            if isinstance(value,Exception):raise value
            return value
        result=wait_ready(observe,self.plan,lambda **r:records.append(r),clock=lambda:clock[0],sleep=lambda n:clock.__setitem__(0,clock[0]+n))
        self.assertTrue(result['controller']['passed'])
        self.assertEqual([r['phase'] for r in records[:2]],['waiting_read_retry']*2)
        self.assertEqual(len(observations),0)

    def test_missing_live_process_without_successful_report_fails_closed(self):
        with self.assertRaises(ValueError):ready(dict(controller=self.controller,birth=None),self.plan)

    def test_launch_payload_has_complete_existing_modules_and_no_local_credentials(self):
        p=dict(native_python='/native/python',asr_python='/asr/python',source='/source',source_commit='a'*40,
               pod_id='owned',upstream='/upstream',paired_out='/workspace/migration/native20',
               speech_out='/workspace/migration/speech20',deadline_utc='2026-10-23T15:27:00+00:00',
               remote_proxy_url='http://127.0.0.1:18789/v1/responses',run_id='scoped',compiler_driver='/driver',
               key='/local/private-key',known_hosts='/local/known-hosts')
        code='import json\n'+launch_payload(p);compile(code,'remote-launch','exec')
        self.assertNotIn('/local/private-key',code)
        self.assertIn('portable.context.native_speech_followup',code)
        command=native_command(p);self.assertEqual(command[command.index('--gpu-mode')+1],'serial-one-gpu')
        self.assertEqual(command[command.index('--candidate-proof')+1],'/upstream/training')


if __name__=='__main__':unittest.main()
