import unittest
from portable.context.native_control_development_wait import ready,native_command,REASONING_STAGES
from portable.context.native_speech_followup import checked_roles


class ReasoningEvaluation(unittest.TestCase):
    def test_new_candidate_requires_every_restored_stage_and_numerical_binding(self):
        p=dict(upstream_pid=1,upstream_birth=2,pod_id='owned',source_commit='new',
               upstream_source_commit='training',candidate_kind='reasoning32')
        c=dict(pid=1,pod_id='owned',source_commit='training',phase='complete_restored_pilot_only',
               passed=True,stages=[dict(name=n,actual_exit=0)for n in REASONING_STAGES],
               reused_numerical=dict(actual_exit=0,controller_sha256='a'*64,numerical_sha256='b'*64))
        self.assertTrue(ready(dict(controller=c,birth=None),p))
        self.assertFalse(ready(dict(controller=c,birth=2),p))
        for bad in (dict(c,stages=c['stages'][:-1]),dict(c,reused_numerical={}),
                    dict(c,source_commit='wrong'),dict(c,phase='upstream_failed_no_dispatch')):
            with self.assertRaises(ValueError):ready(dict(controller=bad,birth=None),p)

    def test_candidate_flag_and_complete_independent_speech_pair(self):
        p=dict(native_python='/python',source='/source',source_commit='new',paired_out='/out',
               asr_python='/asr',deadline_utc='future',pod_id='owned',upstream='/pilot',candidate_kind='reasoning32')
        cmd=native_command(p)
        self.assertIn('--reasoning32-candidate-proof',cmd)
        self.assertNotIn('--candidate-proof',cmd)
        pair=dict(proofs={'baseline520':'base','candidate-reasoning32':'candidate'})
        self.assertEqual(checked_roles(pair,'same','same'),['baseline520','candidate-reasoning32'])
        for value,source in ((pair,'different'),(dict(proofs={'candidate-reasoning32':'candidate'}),'same')):
            with self.assertRaises(ValueError):checked_roles(value,source,'same')


if __name__=='__main__':unittest.main()
