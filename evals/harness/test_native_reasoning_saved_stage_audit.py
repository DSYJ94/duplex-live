"""Failure-mode fixtures for saved-stage inspection; no Torch or GPU execution."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.native_reasoning_saved_stage_audit import (
    SOURCE,CONTROLLER_SOURCE,PINNED,optimizer_rows,terminal_timeout,confined,audit)


class SavedStageAuditTests(unittest.TestCase):
    def setUp(self):
        self.command=['fixture-python','fixture-fit']
        self.command_hash=hashlib.sha256(json.dumps(self.command,separators=(',',':')).encode()).hexdigest()
        self.state=dict(pid=28726,source_commit=SOURCE,controller_commit=CONTROLLER_SOURCE,
            pod_id='8fsk2kouknoiac',phase='failed_preserved_no_retry',passed=False,
            error_type='TimeoutExpired',optimizer_update_limit=8,
            planned_commands=dict(bounded8_training=self.command),stages=[
                dict(name=n,actual_exit=0) for n in ('whole_workset','gradient32','combined_readiness')]+[
                dict(name='bounded8_training',pid=28807,birth=40829227,actual_exit=-15)])

    def terminal(self,controller=None,fit=None):
        with patch('evals.native_reasoning_saved_stage_audit.FIT_COMMAND_SHA256',self.command_hash):
            return terminal_timeout(self.state,controller,fit)

    def test_exact_terminal_timeout_preserves_failed_exit(self):
        self.assertTrue(self.terminal())
        self.assertEqual(self.state['stages'][-1]['actual_exit'],-15)
        self.assertFalse(self.state['passed'])
        self.assertTrue(self.terminal(dict(birth=40829059,state='Z'),dict(birth=40829227,state='Z')))

    def test_live_or_reused_process_handle_is_not_terminal_evidence(self):
        for handle in (dict(birth=40829059,state='R'),dict(birth=999,state='Z')):
            with self.assertRaises(ValueError):self.terminal(controller=handle)
        for handle in (dict(birth=40829227,state='S'),dict(birth=999,state='Z')):
            with self.assertRaises(ValueError):self.terminal(fit=handle)

    def test_other_failure_source_candidate_or_approval_is_rejected(self):
        for key,value in [('phase','bounded8_training'),('error_type','RuntimeError'),
                ('source_commit','b'*40),('controller_commit','b'*40),('pid',999),
                ('pod_id','other'),('passed',True),('optimizer_update_limit',9)]:
            saved=self.state[key];self.state[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.terminal()
            self.state[key]=saved

    def test_original_fit_must_never_be_relabelled_successful(self):
        self.state['stages'][-1]['actual_exit']=0
        with self.assertRaises(ValueError):self.terminal()

    def test_changed_approvals_command_or_handle_rejected(self):
        original=copy.deepcopy(self.state)
        for mutate in (lambda s:s['stages'][0].update(actual_exit=1),
                lambda s:s['stages'][-1].update(birth=999),
                lambda s:s['planned_commands']['bounded8_training'].append('--different'),
                lambda s:s['stages'].pop(1)):
            self.state=copy.deepcopy(original);mutate(self.state)
            with self.assertRaises(ValueError):self.terminal()

    def test_all_eight_receipts_are_required_without_duplicate_or_nonfinite_norm(self):
        rows=[dict(step=i,reasoning32_microbatch=[],gradient_norm_before_clip=1.) for i in range(1,9)]
        text=lambda values:'warning from framework\n'+'\n'.join(json.dumps(r) for r in values)
        self.assertEqual(len(optimizer_rows(text(rows))),8)
        for changed in (rows[:-1],rows+[rows[-1]],list(reversed(rows))):
            with self.assertRaises(ValueError):optimizer_rows(text(changed))
        for norm in (0.,float('nan'),float('inf')):
            rows[0]['gradient_norm_before_clip']=norm
            with self.assertRaises(ValueError):optimizer_rows(text(rows))

    def test_inventory_cannot_escape_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'snapshot';root.mkdir();outside=Path(tmp)/'outside';outside.write_text('fixture')
            (root/'escape').symlink_to(outside)
            for name in ('../outside',str(outside),'escape'):
                with self.assertRaises(ValueError):confined(root.resolve(),name)

    def test_forged_inventory_cannot_substitute_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);name=next(iter(PINNED));f=root/name;f.parent.mkdir(parents=True);f.write_bytes(b'other-state')
            record=dict(path=name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            (root/'inventory.json').write_text(json.dumps(dict(files=[record])))
            with self.assertRaisesRegex(ValueError,'Changed saved-stage artifact'):audit(root)

    def test_duplicate_inventory_paths_cannot_shadow_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'inventory.json').write_text(json.dumps(dict(files=[dict(path='same'),dict(path='same')])))
            with self.assertRaisesRegex(ValueError,'Duplicate inventory'):audit(root)


if __name__=='__main__':unittest.main()
