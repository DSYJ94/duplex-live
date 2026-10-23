import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals import native_reasoning_teacher_cutoff_probe as probe
from evals import native_reasoning_teacher_v7 as recorder
from evals import native_reasoning_stream_v6 as stream
from evals.harness import test_native_reasoning_teacher as artifacts


class OriginalArtifacts(artifacts.TeacherArtifacts):
    implementation=recorder
    capture_stream=staticmethod(stream.capture)


class SingleDiagnosedCapture(unittest.IsolatedAsyncioTestCase):
    async def test_full_failed_audit_required_before_any_new_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            with patch.object(probe.old_quality,'verify',side_effect=ValueError('Whole32 lineage required')),patch.object(probe.recorder,'capture_one')as capture:
                with self.assertRaisesRegex(ValueError,'Whole32'):
                    await probe.probe(p,p,p,p,p/'out',None,p,p)
                capture.assert_not_called();self.assertFalse((p/'out').exists())

    async def test_failed_single_session_is_preserved_without_retry_or_corpus_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);parent=p/'parent';parent.mkdir()
            scene=dict(id=probe.CASE);plan=dict(cases=[scene]);manifest=dict(scenes=[scene])
            async def capture(*args):
                args[3].mkdir();(args[3]/'original-receipt.json').write_text('{}')
                raise ConnectionError('Fixture only')
            with patch.object(probe.old_quality,'verify',return_value=dict(results=[dict(id=probe.CASE,passed=False)])),\
                 patch.object(probe.old_quality,'expected_output',return_value='Okay. Eight tokens. Badge, candle, and poster; total cost 16 tokens; total'),\
                 patch.object(probe.old_quality,'whole_collection',return_value=({},manifest,plan)),\
                 patch.object(probe,'geometry'),patch.object(probe,'sha',return_value=probe.PARENT_SHA),\
                 patch.object(probe.recorder,'capture_one',side_effect=capture)as call:
                value=await probe.probe(p,p,parent,p,p/'out',None,p,p)
            self.assertEqual(call.call_count,1);self.assertEqual(value['status'],'failed_preserved_no_retry')
            self.assertFalse(value['teacher_approved']);self.assertFalse(value['training_approved'])
            self.assertTrue((p/'out'/probe.CASE/'original-receipt.json').exists())


if __name__=='__main__':unittest.main()
