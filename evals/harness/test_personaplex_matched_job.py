import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from training.personaplex import matched_job as j


class FinishedChild:
    pid=1234
    returncode=0
    def poll(self):return self.returncode


class SchedulerOrchestration(unittest.TestCase):
    def test_exact_two_distinct_scheduler_devices_required(self):
        self.assertEqual(j.visible_pair('GPU-a,GPU-b'),['GPU-a','GPU-b'])
        for value in ('','0','0,0','0,1,2','0,'):
            with self.subTest(value=value),self.assertRaises(ValueError):j.visible_pair(value)

    def test_prior_phase_usage_is_verified_against_real_sacct_format(self):
        actual='8216374|brics.u6ada|COMPLETED|327|cpu=4,gres/gpu=1|\n8216374.batch||COMPLETED|327||\n'
        with patch.object(j.subprocess,'check_output',return_value=actual):self.assertEqual(j.prior_usage(),327)
        with patch.object(j.subprocess,'check_output',return_value=actual.replace('|327|','|326|')):
            with self.assertRaises(ValueError):j.prior_usage()

    def children(self,root):
        worker=j.Children(root,'a'*40,root,dict(job_id='100'),None,
            {'personaplex':'GPU-a','context-spanning':'GPU-b'},'/inputs','b'*64)
        worker.ensure=lambda:None
        return worker

    def test_initial_and_final_restore_have_distinct_real_terminal_paths(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);worker=self.children(root);commands=[]
            def launch(command,**kwargs):commands.append(command);return FinishedChild()
            with patch.object(j.subprocess,'Popen',side_effect=launch):
                first=worker.run('personaplex','narrow','restore',root/'initial')
                final=worker.run('personaplex','narrow','restore',root/'final',label='final-restore')
            self.assertNotEqual(first,final)
            a,b=json.loads(first.read_text()),json.loads(final.read_text())
            self.assertNotEqual(a['execution_id'],b['execution_id'])
            self.assertEqual((a['actual_exit_code'],b['actual_exit_code']),(0,0))
            self.assertEqual(a['pid'],1234)
            for command in commands:
                self.assertIn('CUDA_VISIBLE_DEVICES=GPU-a',command)
                self.assertNotIn('CUDA_VISIBLE_DEVICES=GPU-b',command)
                self.assertIn('training.personaplex.matched_native',command)

    def test_actual_child_failure_is_preserved_not_reported_as_a_pass(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);worker=self.children(root);child=FinishedChild();child.returncode=7
            with patch.object(j.subprocess,'Popen',return_value=child):
                with self.assertRaisesRegex(RuntimeError,'Actual native child failed'):
                    worker.run('context-spanning','broad','admit',root/'admit')
            terminal=json.loads((root/'context-spanning-broad-admit-terminal.json').read_text())
            self.assertEqual(terminal['actual_exit_code'],7)
            self.assertEqual(worker.active,{})


if __name__=='__main__':unittest.main()
