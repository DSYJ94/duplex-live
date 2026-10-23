import unittest
from portable.context.native_reasoning_speech_followup import ready


class ActualExitGate(unittest.TestCase):
    def stage(self):
        return dict(phase='complete_verified_failed32_on_local_disk',passed=True,verify_actual_exit=0,
                    verification=dict(complete_cases=32,passed=False,failed_audit_recomputed_on_remote_cpu=True))

    def test_only_same_terminal_supervisor_actual_zero_and_whole_lineage_allow_finalization(self):
        self.assertFalse(ready(dict(birth=1,state='S'),1,{'actual_exit':0},self.stage()))
        self.assertFalse(ready(dict(birth=1,state='S'),1,None,self.stage()))
        self.assertTrue(ready(None,1,{'actual_exit':0},self.stage()))
        self.assertTrue(ready(dict(birth=1,state='Z',exit_status=0),1,{'actual_exit':0},self.stage()))
        self.assertFalse(ready(None,1,{'actual_exit':0},dict(phase='staging_batch')))

    def test_failure_missing_evidence_reused_pid_and_nonzero_supervisor_all_reject(self):
        for actual,exit_value in [(None,None),(dict(birth=2,state='R'),None),
                                  (None,{'actual_exit':1}),
                                  (dict(birth=1,state='Z',exit_status=256),{'actual_exit':0})]:
            with self.subTest(actual=actual),self.assertRaises(ValueError):ready(actual,1,exit_value,self.stage())
        with self.assertRaises(ValueError):
            ready(dict(birth=1,state='R'),1,None,dict(phase='failed_preserved_no_retry'))
