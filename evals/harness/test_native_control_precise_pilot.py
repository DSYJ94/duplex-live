import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from portable.context.native_control_precise_pilot import commands,diagnostic_gate
from evals.native_control_numerical_precise import ORIGINAL_FAILURE_SHA


class SequentialPrecisePilot(unittest.TestCase):
    def test_complete_chain_has_separate_approval_training_and_fresh_restore_processes(self):
        a=SimpleNamespace(python=Path('/python'),out=Path('/out'),root=Path('/root'),source=Path('/source'),
                          cache=Path('/cache'),prepared49=Path('/prepared49'),original_inventory=Path('/original-inventory'),
                          workset_inventory=Path('/workset-inventory'),original_failure=Path('/original-failure'),
                          audit_deadline_utc='deadline')
        with patch('portable.context.native_control_precise_pilot.arguments',return_value=['--all154-old49-control40']):
            rows=commands(a)
        self.assertEqual([r[0] for r in rows],['whole_workset','complete40_precise_numerical',
            'complete40_independent_BF16_gradient','combined_readiness','bounded8_training',
            'independent_restore','strict_proof_verification'])
        for index in (3,4,5):
            args=rows[index][1]
            self.assertIn('--native-audio49-prepared',args);self.assertIn('--native-audio49-source-inventory',args)
            self.assertIn('--all154-old49-control40',args)
        self.assertIn('--native-audio49-preflight-only',rows[3][1])
        self.assertNotIn('--reload',rows[4][1]);self.assertIn('--reload',rows[5][1])
        self.assertEqual(rows[4][1][rows[4][1].index('--out')+1],'/out/training')
        self.assertEqual(rows[5][1][rows[5][1].index('--out')+1],'/out/training')

    def test_diagnostic_only_dispatches_whole_audit_when_all_provenance_is_complete(self):
        d=dict(phase='complete_diagnostic_only',case='native-control3-train-029',optimizer_updates=0,
               training_approved=False,complete40_retested=False,diagnostic_only=True,within_original_threshold=True,
               all_model_parameters_unchanged=True,execution_policy_unchanged=True,
               original_numerical_sha256=ORIGINAL_FAILURE_SHA,
               execution_source_commit='455975c8a882648cb4ae4a18dd478aa36a9b344c',reference_module_sha256='code',
               comparisons=dict(hidden=dict(maximum_absolute_error=.00025),text_logits=dict(maximum_absolute_error=.00049)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'diagnostic.json'
            def digest(p):return 'code' if str(p)=='/attention' else ORIGINAL_FAILURE_SHA
            with patch('portable.context.native_control_precise_pilot.digest',side_effect=digest):
                path.write_text(json.dumps(d));self.assertTrue(diagnostic_gate(path,Path('/original'),Path('/attention')))
                for field,value in [('phase','failed'),('case','another'),('within_original_threshold',False),
                    ('all_model_parameters_unchanged',False),('execution_policy_unchanged',False),
                    ('complete40_retested',True),('training_approved',True),('reference_module_sha256','changed')]:
                    changed=copy.deepcopy(d);changed[field]=value;path.write_text(json.dumps(changed))
                    with self.subTest(field=field),self.assertRaises(ValueError):diagnostic_gate(path,Path('/original'),Path('/attention'))


if __name__=='__main__':unittest.main()
