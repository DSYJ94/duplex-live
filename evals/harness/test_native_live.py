import asyncio
import threading
import unittest
import json
from functools import partial

import numpy as np

from portable.context.native_live import NativeLiveSession
from evals.recovery_worker import run_worker


class Engine:
    frame_size = 1920

    def __init__(self, ret=(0,)):
        self.calls = 0
        self.ret = ret
        self.spans = []

    def step(self, frame):
        index = self.calls
        self.calls += 1
        return dict(text_token=None, agent_pcm=frame, is_ret=index in self.ret)

    def decode_text(self, tokens):return ''

    def inject_context_span(self, text):self.spans.append(text)


def recognized(pcm, rate):
    return dict(input_seconds=len(pcm)/rate, segments=[dict(words=[
        dict(text='Hello', start=.01, end=.05)])])


async def finished(session):
    for _ in range(100):
        if all(task.done() for task in session.pending.values()):return
        await asyncio.sleep(.005)
    raise AssertionError('Fixture worker did not complete')


class LiveSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_worker_protocol_integrates_without_exposing_unavailable_records(self):
        payloads=[]
        def execute(payload):
            payloads.append(payload)
            self.assertEqual(payload['local_worker_state']['dispatch_protocol'],'state-aware')
            self.assertNotIn('lookup_record',[tool['name'] for tool in payload['tools']])
            return dict(id='fixture-only',status='completed',output=[dict(type='function_call',
                name='respond_to_user',call_id='fixture-reply',arguments=json.dumps(dict(
                    route='answer',content='Hello.',consume_pending_speech_ids=[])))])
        engine=Engine();session=NativeLiveSession(engine,recognized,
            worker=partial(run_worker,response_executor=execute))
        session.step(np.zeros(1920,np.float32));await finished(session)
        session.step(np.zeros(1920,np.float32))
        self.assertEqual(engine.spans,['Hello.'])
        self.assertEqual(len(payloads),1)
        session.close()

    async def test_slow_background_answer_does_not_block_frames_or_expire_at_two_seconds(self):
        started, release = threading.Event(), threading.Event()
        def worker(*args, **kwargs):
            self.assertEqual(kwargs['dispatch_protocol'], 'state-aware')
            self.assertTrue(kwargs['scope_read_tools'])
            started.set(); release.wait(timeout=2)
            return dict(route='direct', content='An actual fixture answer.', receipts=[])
        engine = Engine(); session = NativeLiveSession(engine, recognized, worker=worker)
        frame = np.zeros(1920, np.float32)
        try:
            session.step(frame)
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            for _ in range(64):session.step(frame)
            self.assertEqual(engine.calls, 65)
            self.assertFalse(engine.spans)
            release.set(); await finished(session)
            _, events = session.step(frame)
            self.assertEqual(engine.spans, ['An actual fixture answer.'])
            self.assertEqual(events[0]['type'], 'backend_result')
        finally:
            release.set(); session.close()

    async def test_no_request_without_actual_native_ret(self):
        def forbidden(*args, **kwargs):raise AssertionError('Worker must not run')
        engine = Engine(ret=()); session = NativeLiveSession(engine, forbidden, worker=forbidden)
        for _ in range(16):session.step(np.zeros(1920, np.float32))
        await asyncio.sleep(0)
        self.assertFalse(session.pending)
        self.assertEqual(session.requests, 0)
        session.close()

    async def test_close_prevents_old_answer_reaching_reused_engine(self):
        started, release = threading.Event(), threading.Event()
        def worker(*args, **kwargs):
            started.set(); release.wait(timeout=2)
            return dict(route='direct', content='Old conversation.', receipts=[])
        engine = Engine(); old = NativeLiveSession(engine, recognized, worker=worker)
        try:
            old.step(np.zeros(1920, np.float32))
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            old.close(); release.set(); await asyncio.sleep(.02)
            new = NativeLiveSession(engine, recognized, worker=worker)
            new.step(np.zeros(1920, np.float32))
            self.assertFalse(engine.spans)
            self.assertNotEqual(old.identity, new.identity)
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                old.step(np.zeros(1920, np.float32))
            new.close()
        finally:
            release.set(); old.close()

    async def test_actual_cancellation_rejects_late_old_result(self):
        started, release = threading.Event(), threading.Event()
        def worker(context, world, identity, **kwargs):
            if identity.endswith('-1'):
                started.set(); release.wait(timeout=2)
                return dict(route='direct', content='Cancelled old answer.', receipts=[])
            old = next(k for k in world.tasks if k.endswith('-1'))
            receipt = world.cancel(old)
            return dict(route='cancel', content='Cancellation completed.', receipts=[receipt])
        engine = Engine(ret=(0, 1)); session = NativeLiveSession(engine, recognized, worker=worker)
        try:
            session.step(np.zeros(1920, np.float32))
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            session.step(np.zeros(1920, np.float32))
            for _ in range(100):
                if any(v['cancelled'] for v in session.world.tasks.values()):break
                await asyncio.sleep(.005)
            self.assertTrue(any(v['cancelled'] for v in session.world.tasks.values()))
            release.set(); await finished(session)
            session.step(np.zeros(1920, np.float32))
            self.assertEqual(engine.spans, ['Cancellation completed.'])
            self.assertEqual(session.rejected, 1)
        finally:
            release.set(); session.close()

    async def test_worker_failure_is_not_a_fabricated_result_or_stream_crash(self):
        def worker(*args, **kwargs):raise OSError('fixture transport failure')
        engine = Engine(); session = NativeLiveSession(engine, recognized, worker=worker)
        session.step(np.zeros(1920, np.float32)); await finished(session)
        _, events = session.step(np.zeros(1920, np.float32))
        self.assertEqual(engine.calls, 2)
        self.assertEqual(events[0]['route'], 'error')
        self.assertIn('No verified answer', engine.spans[0])
        session.close()


if __name__ == '__main__':unittest.main()
