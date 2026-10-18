import asyncio
from concurrent.futures import Future
import importlib.util
import json
import queue
import struct
import threading
import unittest
import urllib.request
import numpy as np
from serving.hybrid.server import Network


@unittest.skipUnless(importlib.util.find_spec('websockets'),'Requires isolated existing websockets runtime')
class LiveNetworkProtocol(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_restart_aborts_old_capture_before_new_owner(self):
        from websockets.asyncio.server import serve
        from websockets.asyncio.client import connect
        requests=queue.Queue(maxsize=1);network=Network('127.0.0.1',0,'fixture-secret',requests,threading.Event())
        seen=[]
        async def actor():
            for index in range(2):
                item=await asyncio.to_thread(requests.get,True,2);seen.append(item)
                item['output'].event(dict(kind='session_ready'))
                if index==0:
                    while not item['incoming'].aborted:await asyncio.sleep(.005)
                else:
                    while not item['incoming'].finished:await asyncio.sleep(.005)
                    item['output'].event(dict(kind='session_complete'));item['output'].put(None)
                item['future'].set_result(dict(fixture_only=True))
        hello=dict(token='fixture-secret',sample_rate=24000,frame_samples=1920)
        async with serve(network.handler,'127.0.0.1',0,compression=None) as server:
            url='ws://127.0.0.1:'+str(server.sockets[0].getsockname()[1])+'/ws';task=asyncio.create_task(actor())
            async with connect(url,proxy=None) as old:
                await old.send(json.dumps(hello));self.assertEqual(json.loads(await old.recv())['kind'],'session_ready')
                async with connect(url,proxy=None) as unauthenticated:
                    await unauthenticated.send(json.dumps(dict(hello,token='wrong',replace_existing=True)))
                    await asyncio.wait_for(unauthenticated.wait_closed(),1);self.assertEqual(unauthenticated.close_code,1008)
                self.assertFalse(seen[0]['incoming'].aborted)
                async with connect(url,proxy=None) as new:
                    await new.send(json.dumps(dict(hello,replace_existing=True)))
                    self.assertEqual(json.loads(await asyncio.wait_for(new.recv(),2))['kind'],'session_ready')
                    await asyncio.wait_for(old.wait_closed(),1);self.assertEqual(old.close_reason,'Replaced by a new conversation')
                    self.assertTrue(seen[0]['incoming'].aborted);self.assertFalse(seen[1]['incoming'].aborted)
                    await new.send(json.dumps(dict(kind='finish')))
                    self.assertEqual(json.loads(await asyncio.wait_for(new.recv(),1))['kind'],'session_complete')
                await task

    async def test_pause_protocol_rejects_clients_without_retained_buffer_support(self):
        from websockets.asyncio.server import serve
        from websockets.asyncio.client import connect
        requests=queue.Queue(maxsize=1)
        network=Network('127.0.0.1',0,'fixture-secret',requests,threading.Event(),client_pause_protocol=True)
        async with serve(network.handler,'127.0.0.1',0,compression=None) as server:
            url='ws://127.0.0.1:'+str(server.sockets[0].getsockname()[1])+'/ws'
            async with connect(url,proxy=None) as client:
                await client.send(json.dumps(dict(token='fixture-secret',sample_rate=24000,frame_samples=1920)))
                await asyncio.wait_for(client.wait_closed(),timeout=1)
                self.assertEqual(client.close_code,1008);self.assertTrue(requests.empty());self.assertFalse(network.busy)

    async def test_authentication_static_content_and_actual_bidirectional_frame_identity(self):
        from websockets.asyncio.server import serve
        from websockets.asyncio.client import connect
        requests=queue.Queue(maxsize=1);network=Network('127.0.0.1',0,'fixture-secret',requests,threading.Event())
        async with serve(network.handler,'127.0.0.1',0,process_request=network.http,max_size=8192,compression=None,origins=[None]) as server:
            port=server.sockets[0].getsockname()[1];url='ws://127.0.0.1:'+str(port)+'/ws'
            def get():
                with urllib.request.urlopen('http://127.0.0.1:'+str(port)+'/') as response:return response.headers.get_all('Content-Type'),response.read()
            types,body=await asyncio.to_thread(get);self.assertEqual(types,['text/html; charset=utf-8']);self.assertIn(b'Start conversation',body)
            async with connect(url,proxy=None) as client:
                await client.send(json.dumps(dict(token='wrong',sample_rate=24000,frame_samples=1920)))
                await client.wait_closed();self.assertEqual(client.close_code,1008);self.assertTrue(requests.empty())
            async def actor():
                item=await asyncio.to_thread(requests.get,True,2);out=item['output'];incoming=item['incoming'];out.event(dict(kind='session_ready'))
                while True:
                    data=incoming.take()
                    if data is not None:break
                    await asyncio.sleep(.005)
                pcm,received=data;np.testing.assert_array_equal(pcm,np.full(1920,.25,dtype=np.float32));self.assertGreater(received,0)
                out.pcm('speech-3',pcm,1.)
                while not incoming.finished:await asyncio.sleep(.005)
                out.event(dict(kind='session_complete'));out.put(None);item['future'].set_result(dict(fixture_only=True))
            task=asyncio.create_task(actor())
            async with connect(url,proxy=None) as client:
                await client.send(json.dumps(dict(token='fixture-secret',sample_rate=24000,frame_samples=1920,capture_settings={})))
                self.assertEqual(json.loads(await client.recv())['kind'],'session_ready')
                self.assertTrue(network.busy)
                async with connect(url,proxy=None) as invalid:
                    await invalid.send(json.dumps(dict(token='wrong',sample_rate=24000,frame_samples=1920)))
                    await asyncio.wait_for(invalid.wait_closed(),timeout=1);self.assertEqual(invalid.close_code,1008)
                self.assertTrue(network.busy)
                async with connect(url,proxy=None) as concurrent:
                    await concurrent.send(json.dumps(dict(token='fixture-secret',sample_rate=24000,frame_samples=1920)))
                    await asyncio.wait_for(concurrent.wait_closed(),timeout=1);self.assertEqual(concurrent.close_code,1013)
                self.assertTrue(network.busy);self.assertTrue(requests.empty())
                payload=struct.pack('<I',0)+np.full(1920,.25,dtype='<f4').tobytes();await client.send(payload)
                output=await client.recv();self.assertEqual(struct.unpack_from('<II',output),(3,0));self.assertEqual(output[8:],payload[4:])
                await client.send(json.dumps(dict(kind='finish')));self.assertEqual(json.loads(await client.recv())['kind'],'session_complete')
            await task


if __name__=='__main__':unittest.main()
