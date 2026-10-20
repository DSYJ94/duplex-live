import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from evals import native_audio49_chain as chain

class DurableChain(unittest.TestCase):
    def test_uncertain_attempt_is_never_resubmitted(self):
        receipt=dict(jobs={'training':'1'},submission_attempts={'native10':{'status':'submitting'}})
        with patch.object(chain,'validate'),patch.object(chain,'call') as call:
            with self.assertRaises(ValueError):chain.submit(Path('/unused'),receipt,'submit-native','1')
            call.assert_not_called()
    def test_existing_scheduler_job_blocks_next_gpu_submission(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);folder=root/'runs/native-audio49-training-preflight-1';folder.mkdir(parents=True)
            (folder/'audio49-bf16-preflight.json').write_text('{"passed":true,"optimizer_updates":0,"all_parameters_unchanged":true,"source_commit":"source","policy":{"mode":"native520-all49-assistant-audio-smallstep-v1"},"cases":[1,2,3,4]}')
            receipt=dict(jobs={'readiness':'1'},source='source')
            with patch.object(chain,'BASE',root),patch.object(chain,'validate'),patch.object(chain,'call',return_value='1\n') as call:
                with self.assertRaises(ValueError):chain.submit(root/'receipt',receipt,'submit-training','1')
                self.assertEqual(call.call_count,1)
    def test_complete_batch_cannot_submit_after_a_subset_readiness(self):
        import json
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);folder=root/'runs/native-audio49-training-preflight-1';folder.mkdir(parents=True)
            policy=dict(mode='native520-all49-assistant-audio-complete-batch-v1')
            complete=dict(passed=True,optimizer_updates=0,total_components=153,all_parameters_unchanged=True,source_commit='source',policy=policy)
            path=folder/'complete-balanced-preflight.json';path.write_text(json.dumps(complete))
            pre=dict(passed=True,optimizer_updates=0,all_parameters_unchanged=True,source_commit='source',cases=[1,2,3,4],policy=policy,complete_balanced_preflight_sha256=chain.digest(path))
            (folder/'audio49-bf16-preflight.json').write_text(json.dumps(pre))
            with patch.object(chain,'BASE',root),patch.object(chain,'validate'),patch.object(chain,'call') as call:
                with self.assertRaises(AssertionError):chain.submit(root/'receipt',dict(jobs={'readiness':'1'},source='source'),'submit-training','1')
                call.assert_not_called()
    def test_busy_slot_keeps_watcher_alive_without_recording_a_submission(self):
        import json
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'receipt.json';path.write_text(json.dumps(dict(jobs={'readiness':'1'})))
            with patch.object(chain,'validate'),patch.object(chain,'call',return_value='1|COMPLETED|0:0'),patch.object(chain,'submit',side_effect=[chain.SlotBusy('predecessor still visible'),RuntimeError('stop fixture')]) as submit,patch.object(chain.time,'sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError,'stop fixture'):chain.watch(path)
                self.assertEqual(submit.call_count,2);sleep.assert_called_once_with(60)
            self.assertNotIn('submission_attempts',json.loads(path.read_text()))
    def test_preservation_excludes_credential_and_core_paths(self):
        for name in ['runs/a/session-token','keys/openai.env','runs/a/core-344dded','owner-conversations/a.json']:
            self.assertTrue(chain.excluded(Path(name)))
        self.assertFalse(chain.excluded(Path('runs/model/training-state.pt')))
