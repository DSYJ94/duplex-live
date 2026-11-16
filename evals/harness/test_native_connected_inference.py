import unittest
import asyncio
import threading
from portable.context.connected_inference import NativeConnection
import numpy as np


class NativeEngine:
    frame_size=1920
    def __init__(self,tokens):self.tokens=iter(tokens);self.spans=[]
    def step(self,frame):
        token=next(self.tokens,3)
        return dict(agent_pcm=frame,text_token=token,is_ret=token==4)
    def decode_text(self,tokens):return 'Please check the cupboard.'
    def inject_context_span(self,text):self.spans.append(text)


def transcribe(pcm,rate):
    return dict(input_seconds=len(pcm)/rate,segments=[dict(words=[dict(text='cupboard',start=.01,end=.05)])])


class ConnectedNativeTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_ret_reaches_explicit_worker_and_only_completed_result_returns(self):
        seen=[]
        def worker(context,world,identity,**kwargs):
            seen.append((context,kwargs))
            receipt=world.read(identity,'cupboard','marker',kwargs['expected_revision'])
            return dict(route='lookup',content='The marker is red crescent.',receipts=[receipt])
        audit=[]
        engine=NativeEngine([4,3]);connection=NativeConnection(engine,transcribe,worker,records={'cupboard':{'marker':'red crescent'}},audit_sink=audit.append)
        _,events=await connection.step(np.zeros(1920,np.float32))
        self.assertEqual([e['type']for e in events],['ret'])
        self.assertFalse(engine.spans)
        for _ in range(100):
            if all(task.done()for task in connection.session.pending.values()):break
            await asyncio.sleep(.005)
        _,events=await connection.step(np.zeros(1920,np.float32))
        self.assertEqual([e['type']for e in events],['backend_result'])
        self.assertEqual(len(seen),1);self.assertTrue(seen[0][1]['scope_read_tools'])
        self.assertIn('red crescent',engine.spans[0]);self.assertEqual(connection.requests,1)
        delivered=next(r for r in audit if r['type']=='backend_delivered')
        self.assertEqual(delivered['receipts'][0]['value'],'red crescent')
        self.assertTrue(delivered['state']['closed'])
        self.assertIn('context',next(r for r in audit if r['type']=='dispatch_context'))
        connection.close()

    async def test_spoken_promise_does_not_dispatch_without_actual_ret(self):
        def forbidden(*args,**kwargs):raise AssertionError('No native request emitted')
        connection=NativeConnection(NativeEngine([3,3]),forbidden,forbidden)
        for _ in range(2):
            _,events=await connection.step(np.zeros(1920,np.float32));self.assertEqual(events,[])
        self.assertEqual(connection.requests,0);connection.close()

    async def test_background_thread_does_not_block_or_inject_into_closed_connection(self):
        entered,release=threading.Event(),threading.Event()
        def worker(*args,**kwargs):
            entered.set();release.wait(2);return dict(route='direct',content='Retired answer',receipts=[])
        engine=NativeEngine([4,3,3]);connection=NativeConnection(engine,transcribe,worker)
        await connection.step(np.zeros(1920,np.float32))
        self.assertTrue(await asyncio.to_thread(entered.wait,1))
        await connection.step(np.zeros(1920,np.float32))
        self.assertFalse(engine.spans);connection.close();release.set();await asyncio.sleep(.03)
        self.assertFalse(engine.spans)
        with self.assertRaisesRegex(RuntimeError,'closed'):await connection.step(np.zeros(1920,np.float32))

    async def test_supplied_text_is_not_mislabeled_as_audio_recognition(self):
        received=[]
        def forbidden(*args):raise AssertionError('Supplied text must not fake ASR')
        def worker(context,*args,**kwargs):
            received.extend(context);return dict(route='direct',content='Actual local reply.',receipts=[])
        context=lambda seconds,assistant:[dict(role='user',text='What is five minus two?',source='Actual supplied user text in prefix; available before audio playback')]
        connection=NativeConnection(NativeEngine([4,3]),forbidden,worker,context_provider=context)
        await connection.step(np.zeros(1920,np.float32))
        for _ in range(100):
            if all(task.done()for task in connection.session.pending.values()):break
            await asyncio.sleep(.005)
        await connection.step(np.zeros(1920,np.float32))
        self.assertEqual(received[0]['text'],'What is five minus two?')
        self.assertNotIn('start_s',received[0]);self.assertIn('supplied',received[0]['source'])
        connection.close()

    async def test_delivered_work_is_closed_and_cannot_be_cancelled_as_pending(self):
        def worker(*args,**kwargs):return dict(route='direct',content='Finished.',receipts=[])
        connection=NativeConnection(NativeEngine([4,3]),transcribe,worker)
        await connection.step(np.zeros(1920,np.float32))
        for _ in range(100):
            if all(task.done()for task in connection.session.pending.values()):break
            await asyncio.sleep(.005)
        await connection.step(np.zeros(1920,np.float32))
        identity=next(iter(connection.session.world.tasks))
        self.assertTrue(connection.session.world.tasks[identity]['closed'])
        with self.assertRaises(ValueError):connection.session.world.cancel(identity)
        connection.close()

    async def test_external_backend_can_never_be_selected_implicitly(self):
        with self.assertRaisesRegex(TypeError,'worker'):NativeConnection(NativeEngine([]),transcribe)
