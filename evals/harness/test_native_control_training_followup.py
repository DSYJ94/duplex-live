import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portable.context.native_control_training_followup import arguments, gradient_complete, training_environment


class ControlTrainingFollowup(unittest.TestCase):
    def test_exact_old28_and_bounded_recipe_are_carried_to_both_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent=root/'runs/context-training-proof-7510331';parent.mkdir(parents=True)
            rows=[dict(component='/original/'+str(i)) for i in range(28)]
            data=parent/'data.json';data.write_text(json.dumps(dict(components=rows)))
            report=dict(data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),optimizer_steps=520,
                        actual_temporal_preflight='/original/audit')
            (parent/'report.json').write_text(json.dumps(report))
            audit=root/'runs/native-audio49-audit-7572398';audit.mkdir();(audit/'audio49-audit.json').write_text('{}')
            with patch('portable.context.native_control_training_followup.AUDIT_SHA',hashlib.sha256(b'{}').hexdigest()):
                args=arguments(root,root,Path('/cache'),Path('/preflight'),Path('/supervisor'),Path('/evidence'))
                actual=[args[i+1] for i,x in enumerate(args) if x=='--actual-mixed-data']
                self.assertEqual(actual,[x['component'] for x in rows])
                self.assertEqual(args[args.index('--actual-fit-updates')+1],'8')
                self.assertEqual(args[args.index('--native-audio49-step-scale')+1],'0.1')
                self.assertEqual(args[args.index('--control40-development-evidence')+1],'/evidence')
                for flag in ('--native-audio49-complete-batch','--native-audio49-parent-adam',
                             '--native-audio49-stream-auxiliary','--native-v2-warm-ordinary-clock'):
                    self.assertIn(flag,args)
                self.assertNotIn('--resume-training',args)
                rows.pop();data.write_text(json.dumps(dict(components=rows)))
                with self.assertRaises(ValueError):arguments(root,root,root,root,root,root)

    def test_no_key_or_provider_credential_in_training_environment(self):
        original=dict(OPENAI_API_KEY='test-placeholder',OPENAI_BASE_URL='private',AWS_ACCESS_KEY_ID='private',
                      RUNPOD_API_KEY='private',HF_TOKEN='private',PATH='/bin',CPATH='/old')
        actual=training_environment(dict(header='/cuda/include/cuda.h',assembler='/cuda/bin/ptxas'),original,'sha')
        self.assertEqual(actual['PATH'],'/bin');self.assertEqual(actual['DL_TASK_COMMIT'],'sha')
        self.assertFalse(any(n.startswith(('OPENAI_','AWS_','RUNPOD_')) for n in actual))
        self.assertNotIn('HF_TOKEN',actual);self.assertIn('OPENAI_API_KEY',original)

    def test_failed_dead_or_changed_audit_cannot_dispatch_optimizer(self):
        state=dict(pid=10,payload_commit='sha',adopted_numerical_birth=16421263,
                   phase='gradient40',passed=False)
        live=dict(pid=10,birth=50,state='S')
        with patch('portable.context.native_control_training_followup.process',return_value=live):
            self.assertFalse(gradient_complete(state,10,50,'sha'))
            with self.assertRaises(ValueError):gradient_complete(state,10,51,'sha')
            with self.assertRaises(ValueError):gradient_complete(state,10,50,'other')
            with self.assertRaises(RuntimeError):gradient_complete({**state,'phase':'failed'},10,50,'sha')
        with patch('portable.context.native_control_training_followup.process',return_value=None):
            with self.assertRaises(RuntimeError):gradient_complete(state,10,50,'sha')
            complete={**state,'phase':'complete_gradient_only','passed':True,'actual_numerical_exit_code':0,
                      'old_supervisor_retired_after_child_exit':True}
            self.assertTrue(gradient_complete(complete,10,50,'sha'))
            with self.assertRaises(ValueError):gradient_complete({**complete,'passed':False},10,50,'sha')


if __name__=='__main__':unittest.main()
