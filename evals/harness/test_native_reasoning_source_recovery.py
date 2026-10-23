from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from data.synthetic.native_reasoning_delegation import build
from evals.native_reasoning_source import collect, requests_for, sha
from evals.native_reasoning_source_recovery import recover, verify


class SourceRecoveryTests(unittest.TestCase):
    def failed(self, root):
        plan = root/'plan.json'; plan.write_text(json.dumps(build())); calls = []
        def post(*args, **kwargs):
            calls.append(kwargs['json'])
            if len(calls) == 17: raise ConnectionError('Diagnosed fixture read timeout')
            return SimpleNamespace(status_code=200, content=b'original-fixture-'+str(len(calls)).encode(),
                                   headers={'x-request-id':'fixture-original-'+str(len(calls))})
        with redirect_stdout(io.StringIO()), self.assertRaises(ConnectionError):
            collect(plan, root/'failed', None, post=post)
        return root/'failed', plan

    def test_reuses_exact_sixteen_and_exposes_all_forty_one_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); failed, plan=self.failed(root); before=sha(failed/'source-audio.json'); calls=[]
            def post(url, **kwargs):
                self.assertFalse(kwargs['allow_redirects']); self.assertTrue(kwargs['stream'])
                calls.append(kwargs['json'])
                return SimpleNamespace(status_code=200, headers={'x-request-id':'fixture-new-'+str(len(calls))},
                    iter_content=lambda _:iter([b'new-fixture']), close=lambda:None)
            with redirect_stdout(io.StringIO()):r=recover(failed,root/'recovered',None,post=post)
            objects,_=requests_for(json.loads(plan.read_text()))
            self.assertEqual(calls,list(objects.values())[16:]);self.assertEqual(len(calls),24)
            self.assertEqual(r['aggregate_actual_requests'],41);self.assertEqual(len(r['objects']),40)
            self.assertEqual(sha(failed/'source-audio.json'),before)
            for row in r['objects'][:16]:
                self.assertEqual((failed/row['wav']).read_bytes(),(root/'recovered'/row['wav']).read_bytes())
            verify(root/'recovered',require_actual=False)
            with self.assertRaises(ValueError):verify(root/'recovered')
            p=root/'recovered/source-audio.json';bad=json.loads(p.read_text());bad['aggregate_actual_requests']=40
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):verify(root/'recovered',require_actual=False)

    def test_another_failure_preserves_partial_response_and_never_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);failed,_=self.failed(root);calls=[]
            def chunks(_):
                yield b'partial-source-response'
                raise ConnectionError('fixture next transport failure')
            def post(*args,**kwargs):
                calls.append(kwargs['json'])
                return SimpleNamespace(status_code=200,headers={},iter_content=chunks,close=lambda:None)
            with self.assertRaises(ConnectionError):recover(failed,root/'recovered',None,post=post)
            self.assertEqual(len(calls),1)
            self.assertEqual((root/'recovered/failed-response.bin').read_bytes(),b'partial-source-response')
            report=json.loads((root/'recovered/source-audio.json').read_text())
            self.assertEqual(report['aggregate_actual_requests'],18);self.assertFalse(report['complete32'])
            with self.assertRaises(ValueError):verify(root/'recovered',require_actual=False)

    def test_changed_original_wav_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);failed,_=self.failed(root)
            r=json.loads((failed/'source-audio.json').read_text());(failed/r['objects'][0]['wav']).write_bytes(b'changed')
            with self.assertRaises(ValueError):recover(failed,root/'recovered',None,post=lambda *a,**k:None)
            self.assertFalse((root/'recovered').exists())


if __name__=='__main__':unittest.main()
