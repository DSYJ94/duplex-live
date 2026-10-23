"""Reviewer payload/gate fixtures; no actual semantic judgments or API calls."""
import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.synthetic.native_reasoning_delegation import build
from evals.native_reasoning_source import requests_for
from evals import native_reasoning_semantic as semantic


class PayloadBoundary(unittest.TestCase):
    def fixture(self,root,count=32):
        plan=build();objects,scenes=requests_for(plan);manifest=dict(scenes=scenes,objects=[dict(object_id=i,wav_sha256='fixture-wave')for i in objects])
        result=dict(passed=True,results=[dict(id=s['id'],measured_words=[dict(text='Fixture',start=.1,end=.3)],asr_text='Fixture')for s in scenes[:count]])
        batch=root/'batch';batch.mkdir()
        for scene in scenes:
            p=batch/scene['id'];p.mkdir()
            (p/'report.json').write_text(json.dumps(dict(source_starts=[i*24000 for i in range(len(scene['clips']))],observed_until_s=2.)))
            (p/'environment.json').write_text(json.dumps(dict(original_source_bindings=[dict(samples=1000)for _ in scene['clips']])))
            (p/'lifecycle.json').write_text(json.dumps(dict(tasks={})));(p/'events.jsonl').write_text('')
        for case in plan['cases']:case['annotations_not_model_inputs']['oracle_poison']='NEVER_SEND_SCORING_TARGET'
        return batch,manifest,plan,result

    def test_only_actual_speech_and_authored_inputs_leave_the_scoring_boundary(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);batch,m,p,r=self.fixture(root)
            with patch.object(semantic.speech,'verify',return_value=r),patch.object(semantic,'source_manifest',return_value=(m,p)):
                sent=semantic.messages(root,root,batch,root,root,root)
            self.assertEqual(len(sent),32);text=json.dumps(sent)
            self.assertNotIn('NEVER_SEND_SCORING_TARGET',text)
            self.assertNotIn('annotations_not_model_inputs',text)
            self.assertNotIn('expected_final_spoken_words',text)
            self.assertIn('actual_tasks_and_receipts',text)

    def test31_approved_speech_records_never_produce_semantic_inputs(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);batch,m,p,r=self.fixture(root,31)
            with patch.object(semantic.speech,'verify',return_value=r),patch.object(semantic,'source_manifest',return_value=(m,p)):
                with self.assertRaisesRegex(ValueError,'Complete independent32'):semantic.messages(root,root,batch,root,root,root)


class WholeSemanticReview(unittest.IsolatedAsyncioTestCase):
    async def test_no_paid_call_or_output_directory_before_full_speech_approval(self):
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder)
            with patch.object(semantic.speech,'verify',side_effect=ValueError('Speech incomplete')),patch.object(semantic,'request')as request:
                with self.assertRaisesRegex(ValueError,'Speech incomplete'):
                    await semantic.review(p,p,p,p,p,p,p/'out',None)
                request.assert_not_called();self.assertFalse((p/'out').exists())

    async def test_all32_responses_remain_bound_and_missing_or_rewritten_response_rejects(self):
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder);batch=p/'batch';batch.mkdir();(batch/'teacher-collection.json').write_text('{}')
            q=p/'source-quality.json';q.write_text('{}');a=p/'speech.json';a.write_text('{}')
            sent={r['id']:[dict(role='user',content='Artificial reviewer fixture only.')]for r in build()['cases']}
            def request(messages,tools,path):
                call=dict(type='function_call',name=tools[0]['name'],arguments=json.dumps(dict(verdict='pass',reason='Fixture only.')))
                data=dict(id='fixture-'+path.stem,model=semantic.MODEL,status='completed',output=[call])
                path.write_text(json.dumps(data));return data,call,0.
            args=(p,q,batch,p,p,a)
            with patch.object(semantic,'messages',return_value=sent),patch.object(semantic,'request',request),patch.object(semantic,'local_key',return_value='fixture-only'),contextlib.redirect_stdout(io.StringIO()):
                r=await semantic.review(*args,p/'out',None);approval=p/'out/semantic.json'
                self.assertEqual(len(r['results']),32);self.assertFalse(r['training_approved'])
                semantic.verify(*args,approval)
                approval.write_text(json.dumps(dict(r,results=r['results'][:-1])))
                with self.assertRaisesRegex(ValueError,'Complete unchanged'):semantic.verify(*args,approval)
                approval.write_text(json.dumps(r));path=approval.parent/r['results'][0]['response_file']
                data=json.loads(path.read_text());data['output'][0]['arguments']=json.dumps(dict(verdict='fail',reason='Fixture only.'))
                path.write_text(json.dumps(data));r['results'][0]['response_sha256']=semantic.sha(path);approval.write_text(json.dumps(r))
                with self.assertRaisesRegex(ValueError,'Actual semantic response'):semantic.verify(*args,approval)

    async def test_transport_failure_waits_for_other_authorized_calls_without_retry_or_approval(self):
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder);batch=p/'batch';batch.mkdir();(batch/'teacher-collection.json').write_text('{}')
            q=p/'source-quality.json';q.write_text('{}');a=p/'speech.json';a.write_text('{}')
            sent={r['id']:[dict(role='user',content='Artificial fixture only.')]for r in build()['cases']};calls=[]
            def request(messages,tools,path):
                calls.append(path.name)
                if path.name.startswith('native-reasoning4-train-002.'):
                    path.write_text('{"fixture_failure":true}');raise ConnectionError('Fixture only')
                call=dict(type='function_call',name=tools[0]['name'],arguments=json.dumps(dict(verdict='pass',reason='Fixture only.')))
                data=dict(id='fixture-'+path.stem,model=semantic.MODEL,status='completed',output=[call])
                path.write_text(json.dumps(data));return data,call,0.
            with patch.object(semantic,'messages',return_value=sent),patch.object(semantic,'request',request),patch.object(semantic,'local_key',return_value='fixture-only'),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'transport failed'):
                    await semantic.review(p,q,batch,p,p,a,p/'out',None)
            r=json.loads((p/'out/semantic.json').read_text())
            self.assertEqual(len(calls),32);self.assertEqual(len(set(calls)),32)
            self.assertEqual(len(r['results']),31);self.assertEqual(len(r['failures']),1)
            self.assertFalse(r['passed']);self.assertEqual(r['status'],'incomplete_unapproved')


class DerivedSpeechGate(unittest.TestCase):
    def test_explicit_derived_lineage_is_required_and_independently_verified(self):
        from evals import native_reasoning_teacher_speech_v2 as derived
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder);approval=p/'teacher-speech-quality.json';approval.write_text(json.dumps(dict(schema=derived.SCHEMA)))
            with patch.object(derived,'verify',return_value=dict(passed=True))as verify:
                with self.assertRaisesRegex(ValueError,'exact failed audit'):
                    semantic.qualified_speech(p,p,p,p,p,approval)
                verify.assert_not_called()
                semantic.qualified_speech(p,p,p,p,p,approval,p/'failed',p/'plan')
                verify.assert_called_once_with(p,p,p,p,p,p/'failed',p/'plan',p)


if __name__=='__main__':unittest.main()
