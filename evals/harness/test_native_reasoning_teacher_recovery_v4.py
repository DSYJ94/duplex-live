import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.synthetic.native_reasoning_delegation import build
from evals import native_reasoning_teacher_recovery_v4 as recovery
from evals import native_reasoning_teacher_v4 as recorder
from evals import native_reasoning_teacher_quality as quality
from evals.harness import test_native_reasoning_teacher as original
from evals.native_reasoning_stream_v3 import capture


class V4RawArtifacts(original.TeacherArtifacts):
    implementation=recorder
    capture_stream=staticmethod(capture)


class SourceLineage(unittest.TestCase):
    def test_only_identical_per_case_sources_may_reuse_an_old_teacher(self):
        old=dict(scenes=[dict(id='case',clips=[dict(object_id='object',text='Exact words')])],
                 objects=[dict(object_id='object',wav_sha256='original')])
        self.assertEqual(recovery.case_binding(old,copy.deepcopy(old),'case'),old['scenes'][0])
        for mutation in ('words','PCM'):
            new=copy.deepcopy(old)
            if mutation=='words':new['scenes'][0]['clips'][0]['text']='Different words'
            else:new['objects'][0]['wav_sha256']='different'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):recovery.case_binding(old,new,'case')

    def test_quality_v4_cannot_bypass_complete_actual32_verification(self):
        with tempfile.TemporaryDirectory()as folder:
            batch=Path(folder);(batch/'teacher-collection.json').write_text(json.dumps(dict(schema=recovery.SCHEMA)))
            with patch.object(recovery,'verify',side_effect=ValueError('Complete32 required'))as verify:
                with self.assertRaisesRegex(ValueError,'Complete32'):quality.whole_collection(batch,batch,batch,batch,batch)
                verify.assert_called_once_with(batch,batch,batch,batch,batch)


class StopOnFailure(unittest.IsolatedAsyncioTestCase):
    async def test_failed_timing_repair_preserves_all_six_and_does_not_attempt_other25(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);source=root/'source';source.mkdir();q=root/'quality.json';q.write_text('{}')
            parent=root/'parent';parent.mkdir();inventory=root/'inventory.json';inventory.write_text('{}')
            old=dict(recordings=[]);plan=build();manifest=dict(scenes=[dict(id=r['id'],clips=[])for r in plan['cases']])
            for identity in recovery.REUSE:
                d=parent/identity;d.mkdir();(d/'original.wav').write_bytes(identity.encode())
                old['recordings'].append(dict(id=identity,dir=identity,recorder_version='v3',reused_in_v3=False,artifacts={}))
            attempted=[]
            async def capture(*args):
                scene=args[1];folder=args[3];folder.mkdir();attempted.append(scene['id'])
                return dict(id=scene['id'],dir=scene['id'],transport_complete=False,frame_count_diagnostic_pass=False,artifacts={})
            with patch.object(recovery.original,'geometry'),patch.object(recovery,'verify_parent',return_value=(old,manifest,plan,manifest)),\
                 patch.object(recovery.previous,'reference_packing',return_value={}),patch.object(recovery.revised,'capture_one',side_effect=capture):
                r=await recovery.recover(source,q,parent,inventory,root/'out',None,root/'engine',root/'spm')
            self.assertEqual(attempted,[recovery.SENTINEL]);self.assertEqual(r['status'],'incomplete_failed_preserved_no_retry')
            self.assertEqual((r['prior_voice_attempts'],r['maximum_new_sessions'],r['maximum_aggregate_voice_attempts']),(10,26,36))
            self.assertEqual(len(r['new_attempt_order']),26);self.assertFalse(r['training_approved'])
            for identity in recovery.REUSE:
                self.assertEqual((parent/identity/'original.wav').read_bytes(),(root/'out'/identity/'original.wav').read_bytes())
                self.assertEqual((parent/identity/'original.wav').read_bytes(),(root/'out/parent-v3'/identity/'original.wav').read_bytes())
            self.assertTrue(all(x['reused_in_v4']for x in r['recordings']if x['id']in recovery.REUSE))


if __name__=='__main__':unittest.main()
