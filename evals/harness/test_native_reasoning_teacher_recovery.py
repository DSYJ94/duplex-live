"""Recovery provenance and no-progress-on-failure tests; no provider calls."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from data.synthetic.native_reasoning_delegation import build
from evals import native_reasoning_teacher_recovery as recovery
from evals import native_reasoning_teacher_quality as quality


class RecoveryGuards(unittest.TestCase):
    def test_mixed_record_versions_cannot_be_relabelled(self):
        for record in (dict(recorder_version='v1',reused_parent_record=False),dict(recorder_version='v2',reused_parent_record=True)):
            with self.assertRaisesRegex(ValueError,'recording lineage'):recovery.recording_implementation(record)
        self.assertIs(recovery.recording_implementation(dict(recorder_version='v1',reused_parent_record=True)),recovery.original)
        self.assertIs(recovery.recording_implementation(dict(recorder_version='v2',reused_parent_record=False)),recovery.revised)

    def test_new_quality_schema_still_delegates_to_complete_actual_recovery_verifier(self):
        with tempfile.TemporaryDirectory()as folder:
            batch=Path(folder);(batch/'teacher-collection.json').write_text(json.dumps(dict(schema=recovery.SCHEMA)))
            with patch.object(recovery,'verify',side_effect=ValueError('Complete32 required'))as verify:
                with self.assertRaisesRegex(ValueError,'Complete32'):quality.whole_collection(batch,batch,batch,batch,batch)
                verify.assert_called_once_with(batch,batch,batch,batch,batch)


class RecoveryPhaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_two_case_repair_preserves_originals_and_never_starts_26_unattempted(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);source=root/'source';source.mkdir();(source/'source-audio.json').write_text('{}')
            q=root/'quality.json';q.write_text('{}');parent=root/'parent';parent.mkdir();inventory=root/'inventory.json';inventory.write_text('{}')
            old=dict(recordings=[]);plan=build();manifest=dict(scenes=[dict(id=r['id'],clips=[]) for r in plan['cases']])
            original_bytes={}
            for identity in recovery.REUSE:
                d=parent/identity;d.mkdir();p=d/'original.wav';p.write_bytes(('original-'+identity).encode());original_bytes[identity]=p.read_bytes()
                old['recordings'].append(dict(id=identity,dir=identity,transport_complete=True,frame_count_diagnostic_pass=True,
                    artifacts={'original.wav':dict(bytes=len(p.read_bytes()),sha256=recovery.sha(p))}))
            packing=dict(rows=[dict(id=recovery.SENTINEL,reference_frame_margin=5)])
            attempted=[]
            async def capture(source,scene,row,d,*args):
                attempted.append(scene['id']);d.mkdir();return dict(id=scene['id'],dir=scene['id'],transport_complete=False,frame_count_diagnostic_pass=False,artifacts={})
            with patch.object(recovery.original,'geometry'),patch.object(recovery,'verify_parent',return_value=(old,manifest,plan)),patch.object(recovery,'reference_packing',return_value=packing),patch.object(recovery.revised,'capture_one',side_effect=capture):
                r=await recovery.recover(source,q,parent,inventory,root/'out',None,root/'engine',root/'spm')
            self.assertTrue(attempted);self.assertTrue(set(attempted)<=set(recovery.REPAIR))
            self.assertEqual(r['status'],'incomplete_failed_preserved_no_retry');self.assertFalse(r['training_approved'])
            self.assertEqual(r['reused_ids'],recovery.REUSE);self.assertEqual(r['maximum_new_sessions'],28)
            self.assertEqual(r['maximum_aggregate_voice_attempts'],34)
            for identity,data in original_bytes.items():
                self.assertEqual((parent/identity/'original.wav').read_bytes(),data)
                self.assertEqual((root/'out'/identity/'original.wav').read_bytes(),data)
                self.assertEqual((root/'out/parent-v1'/identity/'original.wav').read_bytes(),data)
            self.assertEqual(len(r['new_attempt_order']),28)
            self.assertEqual(r['new_attempt_order'][:3],recovery.REPAIR+[recovery.SENTINEL])


if __name__=='__main__':unittest.main()
