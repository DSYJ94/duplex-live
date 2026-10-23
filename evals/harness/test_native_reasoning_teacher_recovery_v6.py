import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.synthetic.native_reasoning_delegation import build
from evals import native_reasoning_teacher_recovery_v6 as recovery
from evals import native_reasoning_teacher_quality as quality
from evals import native_reasoning_teacher_v6 as recorder
from evals import native_reasoning_stream_v5 as stream
from evals import native_reasoning_stream_v4 as previous_stream
from evals.harness import test_native_reasoning_teacher as artifacts
from evals.harness import test_native_reasoning_stream_v3 as old_stream_tests
from evals.native_reasoning_teacher_lifecycle_review import case_review


class RawArtifacts(artifacts.TeacherArtifacts):
    implementation=recorder
    capture_stream=staticmethod(stream.capture)


class CausalStream(old_stream_tests.CausalStream):
    capture_impl=staticmethod(stream.capture)

    async def test_only_prompt_changes_and_runtime_never_injects_a_request(self):
        new=stream.configuration();old=previous_stream.configuration()
        self.assertEqual({k:v for k,v in new.items()if k!='instructions'},
                         {k:v for k,v in old.items()if k!='instructions'})
        self.assertNotEqual(new['instructions'],old['instructions'])
        for label in ('Backchannel policy:','Interruption policy:','Delegation policy:',
                      'Generate <|SpawnThinking|> when:','Do not generate <|SpawnThinking|> when:'):
            self.assertIn(label,new['instructions'])


class WholeCorpus(unittest.TestCase):
    def test_new_schema_still_demands_every_actual_episode(self):
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder);(p/'teacher-collection.json').write_text(json.dumps(dict(schema=recovery.SCHEMA)))
            with patch.object(recovery,'verify',side_effect=ValueError('Whole32 required'))as verify:
                with self.assertRaisesRegex(ValueError,'Whole32'):quality.whole_collection(p,p,p,p,p)
                verify.assert_called_once_with(p,p,p,p,p)

    def test_answering_arithmetic_does_not_substitute_for_actual_cancellation(self):
        ledger=dict(tasks={'original':dict(phase='wire_sent',cancelled=False,provider_finished_at_s=3.,
                                           wire_write_completed_at_s=9.,receipt=dict(operation='respond_to_user'))})
        row=dict(id='fixture-only',family='cancel-reasoning')
        result=case_review(row,ledger)
        self.assertFalse(result['structural_pass'])
        self.assertIn('Missing actual committed cancel_reasoning for original request',result['reasons'])


class StopOnFailure(unittest.IsolatedAsyncioTestCase):
    async def test_failed_prompt_probe_preserves27_and_never_attempts_other_four(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);source=root/'source';source.mkdir();q=root/'quality.json';q.write_text('{}')
            parent=root/'parent';parent.mkdir();inventory=root/'inventory.json';inventory.write_text('{}')
            old=dict(recordings=[]);plan=build();manifest=dict(scenes=[dict(id=r['id'],clips=[])for r in plan['cases']])
            for identity in recovery.REUSE:
                d=parent/identity;d.mkdir();(d/'original.wav').write_bytes(identity.encode())
                old['recordings'].append(dict(id=identity,dir=identity,artifacts={}))
            attempted=[]
            async def capture(*args):
                scene=args[1];args[3].mkdir();attempted.append(scene['id'])
                return dict(id=scene['id'],dir=scene['id'],transport_complete=False,frame_count_diagnostic_pass=False,artifacts={})
            with patch.object(recovery.original,'geometry'),patch.object(recovery,'verify_parent',return_value=(old,manifest,plan)),\
                 patch.object(recovery,'reference_packing',return_value={}),patch.object(recovery.revised,'capture_one',side_effect=capture):
                r=await recovery.recover(source,q,parent,inventory,root/'out',None,root/'engine',root/'spm')
            self.assertEqual(attempted,[recovery.SENTINEL]);self.assertEqual(r['status'],'incomplete_failed_preserved_no_retry')
            self.assertEqual((r['prior_voice_attempts'],r['maximum_new_sessions'],r['maximum_aggregate_voice_attempts']),(33,5,38))
            self.assertEqual(r['prior_backend_requests'],43);self.assertEqual(len(r['new_attempt_order']),5)
            self.assertFalse(r['training_approved'])
            for identity in recovery.REUSE:
                self.assertEqual((parent/identity/'original.wav').read_bytes(),(root/'out'/identity/'original.wav').read_bytes())
                self.assertEqual((parent/identity/'original.wav').read_bytes(),(root/'out/parent-v5'/identity/'original.wav').read_bytes())


if __name__=='__main__':unittest.main()
