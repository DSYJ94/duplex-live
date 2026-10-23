import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from data.synthetic.native_reasoning_delegation import build
from evals import native_reasoning_source_pacing as pacing
from evals.native_reasoning_source import requests_for, sha


def wav_bytes(value=4000, rate=24000, seconds=.2):
    out = io.BytesIO()
    with wave.open(out, 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes(int(value).to_bytes(2,'little',signed=True)*int(rate*seconds))
    return out.getvalue()


class PacingTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'source'; source.mkdir(); (source/'objects').mkdir()
        plan=build(); objects,scenes=requests_for(plan); rows=[]
        for identity in objects:
            wav=source/'objects'/(identity+'.wav');wav.write_bytes(wav_bytes(3000))
            receipt=wav.with_suffix('.receipt.json');receipt.write_text(json.dumps(dict(original=identity)))
            rows.append(dict(object_id=identity,wav='objects/'+identity+'.wav',wav_sha256=sha(wav),receipt_sha256=sha(receipt)))
        manifest=dict(objects=rows,scenes=scenes,actual_api=False,source_model=next(iter(objects.values()))['model'])
        (source/'source-plan.json').write_text(json.dumps(plan)); (source/'source-audio.json').write_text(json.dumps(manifest))
        quality=root/'quality';quality.mkdir();(quality/'source-quality.json').write_text('{}')
        failed=root/'failed';failed.mkdir();(failed/'native-reasoning4-train-013').mkdir()
        (failed/'teacher-collection.json').write_text('{}');(failed/'native-reasoning4-train-013/report.json').write_text('{}')
        affected=[f'native-reasoning4-train-{i:03d}'for i in (13,14,15)]
        row=next(r for r in rows if r['object_id']==pacing.OBJECT)
        return source,quality/'source-quality.json',failed,(manifest,plan,row,affected)

    def response(self, data=None, status=200, identity='fixture-provider'):
        return SimpleNamespace(status_code=status,headers={'x-request-id':identity},
                               iter_content=lambda _:iter([wav_bytes() if data is None else data]),close=lambda:None)

    def test_payload_changes_only_pacing_and_retains_exact_matched_words_and_voice(self):
        plan=build();before=copy.deepcopy(requests_for(plan)[0][pacing.OBJECT]);after=pacing.payload_for(plan)
        self.assertEqual({k:v for k,v in after.items()if k!='instructions'},
                         {k:v for k,v in before.items()if k!='instructions'})
        self.assertNotEqual(pacing.canonical_hash(after),pacing.OBJECT)
        self.assertNotEqual(after['instructions'],before['instructions'])

    def test_fixture_take_stays_nonactual_and_all40_preserve_original_parents(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);source,quality,failed,binding=self.fixture(root)
            with patch.object(pacing,'preflight',return_value=binding):
                value=pacing.collect(source,quality,failed,root/'take',None,post=lambda *a,**k:self.response())
                self.assertFalse(value['actual_api']);self.assertFalse(value['whole40_source_quality_approved'])
                r=pacing.prepare(source,quality,root/'take',root/'v4',require_actual=False)
                self.assertEqual(len(r['objects']),40);self.assertEqual(r['reused_objects'],39)
                self.assertEqual(r['aggregate_actual_requests'],43)
                self.assertEqual(sha(source/'source-audio.json'),sha(root/'v4/parent-v3/source-audio.json'))
                pacing.verify(root/'v4',require_actual=False)
                with self.assertRaises(ValueError):pacing.verify(root/'v4')

    def test_forged_payload_alias_changed_words_originals_and_receipts_reject(self):
        for mutation in ('alias','words','original','receipt'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as folder:
                root=Path(folder);source,quality,failed,binding=self.fixture(root)
                with patch.object(pacing,'preflight',return_value=binding):
                    pacing.collect(source,quality,failed,root/'take',None,post=lambda *a,**k:self.response())
                    if mutation=='alias':
                        p=root/'take/take.json';v=json.loads(p.read_text());v['actual_payload_sha256']=pacing.OBJECT;p.write_text(json.dumps(v))
                    elif mutation=='words':
                        p=root/'take/request.json';v=json.loads(p.read_text());v['input']='Different words';p.write_text(json.dumps(v))
                    elif mutation=='original':(root/'take/original.wav').write_bytes(wav_bytes(9000))
                    else:
                        p=root/'take/replacement.receipt.json';v=json.loads(p.read_text());v['sha256']='forged';p.write_text(json.dumps(v))
                    with self.assertRaises(ValueError):pacing.verify_take(source,quality,root/'take',require_actual=False)

    def test_incomplete_corpus_and_unlisted_artifacts_are_not_accepted(self):
        for mutation in ('subset','extra'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as folder:
                root=Path(folder);source,quality,failed,binding=self.fixture(root)
                with patch.object(pacing,'preflight',return_value=binding):
                    pacing.collect(source,quality,failed,root/'take',None,post=lambda *a,**k:self.response())
                    pacing.prepare(source,quality,root/'take',root/'v4',require_actual=False)
                    if mutation=='subset':
                        p=root/'v4/source-audio.json';v=json.loads(p.read_text());v['objects'].pop();p.write_text(json.dumps(v))
                    else:(root/'v4/unlisted.json').write_text('{}')
                    with self.assertRaises(ValueError):pacing.verify(root/'v4',require_actual=False)

    def test_failed_or_unreceipted_transport_preserves_raw_bytes_without_retry(self):
        for status,identity in ((503,'fixture-provider'),(200,None)):
            with self.subTest(status=status,identity=identity),tempfile.TemporaryDirectory()as folder:
                root=Path(folder);source,quality,failed,binding=self.fixture(root);calls=[]
                def post(*args,**kwargs):
                    calls.append(kwargs);return self.response(status=status,identity=identity)
                with patch.object(pacing,'preflight',return_value=binding),self.assertRaises(RuntimeError):
                    pacing.collect(source,quality,failed,root/'take',None,post=post)
                self.assertEqual(len(calls),1);self.assertIs(calls[0]['allow_redirects'],False)
                self.assertEqual(json.loads((root/'take/take.json').read_text())['phase'],'failed_preserved_no_retry')
                self.assertTrue((root/'take/response.partial').exists())

    def test_rate_clipping_and_overlong_take_reject_without_waveform_edits(self):
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'take.wav'
            for data in (wav_bytes(32767),wav_bytes(rate=16000),wav_bytes(seconds=40.01)):
                path.write_bytes(data)
                with self.assertRaises(ValueError):pacing.waveform(path)
                self.assertEqual(path.read_bytes(),data)


if __name__=='__main__':unittest.main()
