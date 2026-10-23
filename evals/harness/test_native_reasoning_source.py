import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from data.synthetic.native_reasoning_delegation import build
from evals.native_reasoning_source import collect,requests_for,verify,ENDPOINT


class NativeReasoningSource(unittest.TestCase):
    def test_only_actual_authored_user_words_enter_forty_requests_for_fifty_six_clips(self):
        p=build();objects,scenes=requests_for(p)
        self.assertEqual(len(objects),40);self.assertEqual(len(scenes),32)
        self.assertEqual(sum(len(s['clips']) for s in scenes),56)
        for i in range(8):
            self.assertEqual(len({scenes[4*i+j]['clips'][0]['object_id'] for j in (1,2,3)}),1)
        for payload in objects.values():
            self.assertEqual(set(payload),{'model','input','voice','response_format','instructions'})
            self.assertNotIn('oracle',json.dumps(payload));self.assertNotIn('expected_',json.dumps(payload))
        bad=copy.deepcopy(p);bad['cases'][0]['inference_inputs']['answer']='do not transmit'
        with self.assertRaises(ValueError):requests_for(bad)

    def test_fixture_generation_cannot_pass_real_source_verification_and_mutations_reject(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);plan=root/'plan.json';plan.write_text(json.dumps(build()));calls=[]
            def post(url,**kwargs):
                self.assertEqual(url,ENDPOINT);calls.append(kwargs['json'])
                return SimpleNamespace(status_code=200,content=b'fixture-wave-not-quality-approved',headers={'x-request-id':'fixture-'+str(len(calls))})
            with redirect_stdout(io.StringIO()):r=collect(plan,root/'source',None,post=post)
            self.assertEqual(len(calls),40);self.assertFalse(r['training_approved'])
            with self.assertRaises(ValueError):verify(root/'source')
            verify(root/'source',require_actual=False)
            wav=root/'source'/r['objects'][0]['wav'];wav.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify(root/'source',require_actual=False)

    def test_first_transport_failure_is_preserved_without_retry_or_subset_approval(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);plan=root/'plan.json';plan.write_text(json.dumps(build()));calls=[]
            def post(*args,**kwargs):
                calls.append(kwargs['json']);return SimpleNamespace(status_code=429)
            with self.assertRaises(RuntimeError):collect(plan,root/'source',None,post=post)
            self.assertEqual(len(calls),1)
            r=json.loads((root/'source/source-audio.json').read_text())
            self.assertEqual(r['phase'],'failed_preserved_no_retry');self.assertFalse(r['complete32'])
            with self.assertRaises(ValueError):verify(root/'source',require_actual=False)
            with self.assertRaises(FileExistsError):collect(plan,root/'source',None,post=post)


if __name__=='__main__':unittest.main()
