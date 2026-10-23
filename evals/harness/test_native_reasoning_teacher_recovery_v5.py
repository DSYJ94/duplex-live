import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.synthetic.native_reasoning_delegation import build
from evals import native_reasoning_teacher_recovery_v5 as recovery
from evals import native_reasoning_teacher_quality as quality
from evals import native_reasoning_teacher_v5 as recorder
from evals import native_reasoning_stream_v4 as stream
from evals import native_reasoning_stream_v3 as previous_stream
from evals.harness import test_native_reasoning_teacher as artifacts
from evals.harness import test_native_reasoning_stream_v3 as old_stream_tests


class RawArtifacts(artifacts.TeacherArtifacts):
    implementation=recorder
    capture_stream=staticmethod(stream.capture)


class CausalStream(old_stream_tests.CausalStream):
    capture_impl=staticmethod(stream.capture)

    async def test_voice_configuration_is_unchanged_by_backend_action_repair(self):
        self.assertEqual(stream.configuration(),previous_stream.configuration())


class WholeCorpus(unittest.TestCase):
    def test_new_schema_cannot_bypass_complete_actual32_readback(self):
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder);(p/'teacher-collection.json').write_text(json.dumps(dict(schema=recovery.SCHEMA)))
            with patch.object(recovery,'verify',side_effect=ValueError('Whole32 required'))as verify:
                with self.assertRaisesRegex(ValueError,'Whole32'):quality.whole_collection(p,p,p,p,p)
                verify.assert_called_once_with(p,p,p,p,p)


class StopOnFailure(unittest.IsolatedAsyncioTestCase):
    async def test_failed_action_repair_retains21_originals_and_never_attempts_other10(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);source=root/'source';source.mkdir();q=root/'quality.json';q.write_text('{}')
            parent=root/'parent';parent.mkdir();inventory=root/'inventory.json';inventory.write_text('{}')
            component=root/'component';component.mkdir();review=root/'review.json';review.write_text('{}')
            old=dict(recordings=[]);plan=build();manifest=dict(scenes=[dict(id=r['id'],clips=[])for r in plan['cases']])
            for identity in recovery.REUSE:
                d=parent/identity;d.mkdir();(d/'original.wav').write_bytes(identity.encode())
                old['recordings'].append(dict(id=identity,dir=identity,artifacts={}))
            attempted=[]
            async def capture(*args):
                scene=args[1];args[3].mkdir();attempted.append(scene['id'])
                return dict(id=scene['id'],dir=scene['id'],transport_complete=False,frame_count_diagnostic_pass=False,artifacts={})
            with patch.object(recovery.original,'geometry'),patch.object(recovery,'verify_parent',return_value=(old,manifest,plan)),\
                 patch.object(recovery,'component_gate'),patch.object(recovery.previous.previous,'reference_packing',return_value={}),\
                 patch.object(recovery.revised,'capture_one',side_effect=capture):
                r=await recovery.recover(source,q,parent,inventory,component,review,root/'out',None,root/'engine',root/'spm')
            self.assertEqual(attempted,[recovery.SENTINEL]);self.assertEqual(r['status'],'incomplete_failed_preserved_no_retry')
            self.assertEqual((r['prior_voice_attempts'],r['maximum_new_sessions'],r['maximum_aggregate_voice_attempts']),(26,11,37))
            self.assertEqual(r['prior_backend_requests'],33);self.assertEqual(len(r['new_attempt_order']),11)
            self.assertFalse(r['training_approved'])
            for identity in recovery.REUSE:
                self.assertEqual((parent/identity/'original.wav').read_bytes(),(root/'out'/identity/'original.wav').read_bytes())
                self.assertEqual((parent/identity/'original.wav').read_bytes(),(root/'out/parent-v4'/identity/'original.wav').read_bytes())


if __name__=='__main__':unittest.main()
