from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
import wave

from data.synthetic.native_reasoning_delegation import build
from evals.native_reasoning_source import collect, requests_for, sha
from evals.native_reasoning_source_recovery import recover
from evals.native_reasoning_source_repair import prepare, verify, OBJECT, OLD_QUALITY_CODE


def wav_bytes(value):
    out=io.BytesIO()
    with wave.open(out,'wb')as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000)
        f.writeframes(int(value).to_bytes(2,'little',signed=True)*240)
    return out.getvalue()


class SourceRepairTests(unittest.TestCase):
    def fixture(self,root):
        plan=root/'plan.json';plan.write_text(json.dumps(build()));objects,_=requests_for(build());calls=[]
        def initial(*args,**kwargs):
            calls.append(kwargs['json'])
            if len(calls)==17:raise ConnectionError('fixture read timeout')
            data=wav_bytes(-32767 if calls[-1]==objects[OBJECT] else 1000)
            return SimpleNamespace(status_code=200,content=data,headers={'x-request-id':'original-'+str(len(calls))})
        with redirect_stdout(io.StringIO()),self.assertRaises(ConnectionError):collect(plan,root/'failed',None,post=initial)
        def post(*args,**kwargs):
            return SimpleNamespace(status_code=200,headers={'x-request-id':'new-'+str(len(calls))},
                iter_content=lambda _:iter([wav_bytes(1000)]),close=lambda:None)
        with redirect_stdout(io.StringIO()):parent=recover(root/'failed',root/'parent',None,post=post)
        quality=root/'quality.json';quality.write_text(json.dumps(dict(code_sha256=OLD_QUALITY_CODE,passed=False,
            source_manifest_sha256=sha(root/'parent/source-audio.json'),results=[dict(object_id=x['object_id'],
                peak=.999969482421875 if x['object_id']==OBJECT else .1)for x in parent['objects']])))
        take=root/'take';take.mkdir();payload=objects[OBJECT];(take/'request.json').write_text(json.dumps(payload))
        old=next(x for x in parent['objects']if x['object_id']==OBJECT)
        shutil.copyfile(root/'parent'/old['wav'],take/'original-rejected.wav')
        shutil.copyfile((root/'parent'/old['wav']).with_suffix('.receipt.json'),take/'original-rejected.receipt.json')
        (take/'replacement.wav').write_bytes(wav_bytes(4000))
        receipt=dict(request_id='replacement-fixture',source_model=payload['model'],voice=payload['voice'],
            input_text=payload['input'],instructions=payload['instructions'],sha256=sha(take/'replacement.wav'),
            attempt=1,actual_api=False,request_payload_sha256=OBJECT,response_bytes=(take/'replacement.wav').stat().st_size)
        (take/'replacement.receipt.json').write_text(json.dumps(receipt))
        (take/'take.json').write_text(json.dumps(dict(schema='dl-native-reasoning32-single-diagnosed-source-take-v1',
            phase='collected_unapproved_new_take',object_id=OBJECT,request_payload_sha256=OBJECT,
            maximum_requests=1,actual_requests=1,aggregate_prior_requests=41,aggregate_requests=42,
            source_manifest_sha256=sha(root/'parent/source-audio.json'),original_quality_sha256=sha(quality),
            approved_utterance_or_voice_or_instructions_changed=False,automatic_retry=False,training_approved=False,
            whole40_source_quality_approved=False,wav_sha256=sha(take/'replacement.wav'),receipt_sha256=sha(take/'replacement.receipt.json'))))
        return root/'parent',take,quality

    def test_all40_and_rejected_parent_are_preserved_but_fixture_cannot_be_approved(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);parent,take,quality=self.fixture(root);before=sha(parent/'source-audio.json')
            r=prepare(parent,take,quality,root/'complete',require_actual=False)
            self.assertEqual(len(r['objects']),40);self.assertEqual(r['reused_objects'],39)
            self.assertEqual(r['aggregate_actual_requests'],42);self.assertEqual(r['actual_requests'],0)
            self.assertEqual(sha(parent/'source-audio.json'),before)
            verify(root/'complete',require_actual=False)
            with self.assertRaises(ValueError):verify(root/'complete')
            changed=next(x for x in r['objects']if x['object_id']==OBJECT)
            (root/'complete'/changed['wav']).write_bytes(wav_bytes(5000))
            with self.assertRaises(ValueError):verify(root/'complete',require_actual=False)

    def test_word_or_peak_changes_cannot_be_disguised_as_the_one_diagnosed_take(self):
        for mutation in ('word','peak'):
            with tempfile.TemporaryDirectory()as folder:
                root=Path(folder);parent,take,quality=self.fixture(root)
                if mutation=='word':
                    p=take/'request.json';r=json.loads(p.read_text());r['input']='Unapproved different words';p.write_text(json.dumps(r))
                else:(take/'replacement.wav').write_bytes(wav_bytes(32767))
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    prepare(parent,take,quality,root/'complete',require_actual=False)
                self.assertFalse((root/'complete').exists())


if __name__=='__main__':unittest.main()
