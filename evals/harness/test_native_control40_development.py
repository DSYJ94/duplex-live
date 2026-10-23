import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portable.context.native_development import command,proofs,restored_control40
from evals.native_control_stage import SCHEMA


class Control40Development(unittest.TestCase):
    def test_explicit_new_checkpoint_preserves_complete_old_comparator(self):
        old=proofs();new=proofs(Path('/durable/control40/training'))
        self.assertEqual(new['baseline520'],old['baseline520'])
        self.assertEqual(list(new),['baseline520','candidate-control40'])
        case=dict(source='old',scenario='recovery-validation-000',kind='private',backend='native-asr')
        cmd=command(case,new['candidate-control40'],Path('/out'),Path('/isolated-asr'))
        self.assertEqual(cmd[cmd.index('--adapter-proof')+1],'/durable/control40/training')
        self.assertNotIn('--train-history',cmd)
        self.assertNotIn('--ret-logit-bias',cmd)
        self.assertNotIn('--native-runtime-profile',cmd)

    def test_same_process_or_unapproved_restore_cannot_start_native_review(self):
        report=dict(schema=SCHEMA,updates_this_run=8,adam_parameter_steps_after=[528],training_pid=100)
        reload=dict(pid=101,parent_adam_parameter_steps_restored=[528],optimizer_and_rng_independently_compared_exact=True)
        with tempfile.TemporaryDirectory() as directory:
            proof=Path(directory);(proof/'report.json').write_text(json.dumps(report))
            with patch('evals.native_control_stage.verify_checkpoint',return_value=({'proof':'identity'},report)) as strict:
                (proof/'reload-report.json').write_text(json.dumps(reload))
                r=restored_control40(proof)
                self.assertTrue(r['preexisting_fresh_full_cuda_restore_verified'])
                self.assertFalse(r['new_restore_executed_by_this_check'])
                strict.assert_called_once_with(proof)
                for bad in ({**reload,'pid':100},{**reload,'parent_adam_parameter_steps_restored':[520]},
                            {**reload,'optimizer_and_rng_independently_compared_exact':False}):
                    (proof/'reload-report.json').write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):restored_control40(proof)
            with patch('evals.native_control_stage.verify_checkpoint',side_effect=ValueError('Missing complete40 approval')):
                with self.assertRaisesRegex(ValueError,'complete40'):restored_control40(proof)


if __name__=='__main__':unittest.main()
