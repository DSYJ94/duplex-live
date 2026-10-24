import copy
import json
from pathlib import Path
import unittest

from portable.context.native_reasoning_readiness_followup import eligible, followup_commands, PREFIX


class ReadinessFollowup(unittest.TestCase):
    def setUp(self):
        self.state = dict(pid=8514, source_commit='a'*40, pod_id='owned',
            phase='failed_preserved_no_retry', passed=False, optimizer_update_limit=8,
            error_type='TimeoutError', error='Insufficient bounded headroom for bounded8_training',
            stages=[dict(name=n, actual_exit=0, pid=i) for i,n in enumerate(PREFIX)],
            reused_numerical=dict(actual_exit=0, controller_sha256='b'*64, numerical_sha256='c'*64))

    def check(self, state=None, live=None):
        return eligible(state or self.state, live, 8514, 1234, 'a'*40, 'owned')

    def test_only_actual_terminal_pre_fit_headroom_refusal_is_reusable(self):
        self.check()
        self.check(live=dict(state='Z', birth=1234))
        for value in [dict(state='R',birth=1234),dict(state='S',birth=1234),dict(state='Z',birth=5678)]:
            with self.subTest(value=value), self.assertRaises(ValueError): self.check(live=value)

    def test_failures_and_any_started_training_are_not_retried(self):
        for changed in [dict(error='combined_readiness failed; no retry or downstream dispatch'),
                        dict(error_type='RuntimeError'),dict(phase='complete_restored_pilot_only'),
                        dict(stages=self.state['stages']+[dict(name='bounded8_training',actual_exit=0)]),
                        dict(stages=[dict(name=n,actual_exit=1) for n in PREFIX]),
                        dict(stages=self.state['stages'][:-1])]:
            with self.subTest(changed=changed),self.assertRaises(ValueError):self.check(dict(self.state,**changed))

    def test_wrong_source_pod_identity_or_numerical_binding_fails(self):
        for changed in [dict(pid=9),dict(source_commit='b'*40),dict(pod_id='other'),
                        dict(optimizer_update_limit=32),dict(reused_numerical={})]:
            with self.subTest(changed=changed),self.assertRaises(ValueError):self.check(dict(self.state,**changed))

    def commands(self):
        recipe=json.loads((Path(__file__).resolve().parents[2]/'docs/NATIVE-REASONING32-PLANNED-RECIPE-2026-10-24.json').read_text())
        fit=recipe['execution']['fit_command']
        return dict(planned_commands=dict(bounded8_training=fit,independent_restore=fit+['--reload']))

    def test_every_original_fit_argument_preserved_except_new_output(self):
        value=self.commands();before=copy.deepcopy(value)
        old=Path('/workspace/migration/native-reasoning32-retained-pilot-v17');new=old.with_name('native-reasoning32-retained-pilot-v18')
        result=dict(followup_commands(value,old,new))
        expected=list(value['planned_commands']['bounded8_training']);expected[expected.index('--out')+1]=str(new/'training')
        self.assertEqual(result['bounded8_training'],expected)
        self.assertEqual(result['independent_restore'],expected+['--reload'])
        self.assertEqual(value,before)
        self.assertIn(str(new/'training'),result['strict_proof'][-1])
        self.assertIn(str(old/'combined-readiness'),result['bounded8_training'])

    def test_changed_update_count_readiness_output_or_restore_rejected(self):
        old=Path('/workspace/migration/native-reasoning32-retained-pilot-v17');new=Path('/new')
        for flag,replacement in [('--actual-fit-updates','32'),('--native-audio49-preflight','/partial'),('--out','/other')]:
            value=self.commands()
            for c in value['planned_commands'].values():c[c.index(flag)+1]=replacement
            with self.subTest(flag=flag),self.assertRaises(ValueError):followup_commands(value,old,new)
        value=self.commands();value['planned_commands']['independent_restore'].append('--resume-training')
        with self.assertRaises(ValueError):followup_commands(value,old,new)


if __name__=='__main__':unittest.main()
