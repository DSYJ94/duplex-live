import asyncio
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evals import native_reasoning_semantic_prepared as prepared


class PreparedBoundaryTests(unittest.TestCase):
    def test_changed_payload_or_weakened_readback_cannot_enter_paid_review(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);batch=root/'batch';batch.mkdir();(batch/'teacher-collection.json').write_text('{}')
            approval=root/'speech.json';approval.write_text('{}');quality=root/'source-quality.json';quality.write_text('{}')
            c=SimpleNamespace(source_quality=quality)
            sent={identity:[dict(role='system',content=prepared.semantic.SYSTEM),dict(role='user',content='Synthetic fixture only.')]
                  for identity in prepared.mixed.IDS}
            proof=dict(complete_cases=32,whole_speech_passed=True,all_original_transports_verified=True,
                all_speech_decisions_recomputed_twice=True,unchanged_input_snapshot=True,training_approved=False,
                report_sha256=prepared.sha(approval),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
            proof_path=root/'independent-readback.json';proof_path.write_text(json.dumps(proof))
            target=root/'prepared.json'
            with patch.object(prepared,'checked_messages',return_value=sent),patch.object(prepared.mixed,'source_pins',return_value={'fixture':'unchanged'}):
                prepared.prepare(c,batch,approval,target)
                self.assertEqual(prepared.validate(c,batch,approval,target),sent)
                original=target.read_text();value=json.loads(original)
                value['messages'][prepared.mixed.IDS[0]][1]['content']='Injected scoring or owner content'
                target.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'Exact source-bound'):
                    prepared.validate(c,batch,approval,target)
                target.write_text(original);proof['all_speech_decisions_recomputed_twice']=False
                proof_path.write_text(json.dumps(proof));value=json.loads(original)
                value['verified_readback_sha256']=prepared.sha(proof_path);target.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'Independent complete32'):
                    prepared.validate(c,batch,approval,target)

    def test_incomplete_speech_has_no_paid_calls_or_output_directory(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder)
            with (patch.object(prepared,'checked_messages',side_effect=ValueError('Complete32 missing')),
                  patch.object(prepared.semantic,'review_messages')as review):
                with self.assertRaisesRegex(ValueError,'Complete32 missing'):
                    asyncio.run(prepared.review(None,root,root,root,root/'out',None))
                review.assert_not_called();self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
