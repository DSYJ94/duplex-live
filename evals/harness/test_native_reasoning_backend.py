import asyncio
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from evals.native_reasoning_backend import request
from evals.native_reasoning_lifecycle import NativeReasoningLifecycle


def reply(identity,operation='respond_to_user',**arguments):
    return dict(status='completed',id=identity,model='fixture-only',output=[dict(
        type='function_call',name=operation,arguments=json.dumps(arguments))])


def response(value,status=200):
    return SimpleNamespace(status_code=status,headers={'x-request-id':'fixture-header'},
        iter_content=lambda _:iter([json.dumps(value).encode()]),close=lambda:None)


class ReasoningBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_output_cannot_start_or_overwrite_a_request(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);world=NativeReasoningLifecycle(lambda:1.,0.);calls=[]
            await world.reserve_native_request('one',[dict(speaker='user',text='Question.',received_at_s=.5)])
            with self.assertRaises(FileExistsError):
                await request(world,'one','fixture-only',root,post=lambda *a,**k:calls.append(k))
            self.assertEqual(calls,[]);self.assertEqual(world.tasks['one']['phase'],'reserved')

    async def test_pending_HTTP_does_not_block_cancel_and_late_body_is_retained(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);world=NativeReasoningLifecycle(lambda:10.,0.);entered=asyncio.Event();release=threading.Event()
            context=[dict(speaker='user',text='Original actual problem.',received_at_s=1.)]
            await world.reserve_native_request('original',context)
            def old_post(*args,**kwargs):
                self.assertFalse(kwargs['allow_redirects']);self.assertNotIn('Authorization',kwargs['json'])
                if not release.wait(5):raise TimeoutError('fixture parent did not release')
                return response(reply('old',content='Stale answer.'))
            pending=asyncio.create_task(request(world,'original','fixture-only',root/'old',post=old_post,started=entered))
            await asyncio.wait_for(entered.wait(),2)
            try:
                await world.reserve_native_request('cancel',context+[dict(speaker='user',text='Cancel it.',received_at_s=2.)])
                receipt=await asyncio.wait_for(request(world,'cancel','fixture-only',root/'cancel',post=lambda *a,**k:
                    response(reply('cancel','cancel_reasoning',task_ids=['original'],content='Delivery cancelled.'))),2)
                self.assertEqual(receipt['retired_requests'],[dict(task_id='original',revision=1)])
                self.assertFalse(receipt['actual_transport']);self.assertFalse(receipt['remote_execution_aborted'])
            finally:release.set()
            self.assertIsNone(await asyncio.wait_for(pending,2))
            self.assertEqual(world.tasks['original']['response']['id'],'old')
            trace=json.loads((root/'old/transport.json').read_text())
            self.assertEqual(trace['phase'],'stale_completion_preserved');self.assertFalse(trace['actual_provider_transport'])
            self.assertTrue((root/'old/response.json').is_file())
            self.assertNotIn('fixture-only-no-credential',(root/'old/transport.json').read_text())

    async def test_HTTP_failure_preserves_body_without_fallback_or_retry(self):
        with tempfile.TemporaryDirectory()as folder:
            root=Path(folder);world=NativeReasoningLifecycle(lambda:1.,0.);calls=[]
            await world.reserve_native_request('bad',[dict(speaker='user',text='Actual question.',received_at_s=.5)])
            def post(*args,**kwargs):calls.append(kwargs);return response(dict(error='fixture unavailable'),503)
            with self.assertRaises(RuntimeError):await request(world,'bad','fixture-only',root/'failed',post=post)
            self.assertEqual(len(calls),1);self.assertEqual(world.tasks['bad']['phase'],'transport_failed_preserved')
            self.assertEqual(json.loads((root/'failed/response.json').read_text()),dict(error='fixture unavailable'))
            with self.assertRaises(ValueError):await world.publish('bad',lambda _:None)


if __name__=='__main__':unittest.main()
