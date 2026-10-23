"""Paced protocol fixtures; never actual provider or native-behavior evidence."""
import asyncio
import base64
import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np

from evals.native_control_teacher import atomic_json
from evals.native_reasoning_stream import capture,configuration,RESPONSE_WINDOW_S


class Socket:
    def __init__(self,*,control=False,delegate=True,ack=True,pending_original=True,provider_delay_s=0.,ack_delay_s=0.):
        self.queue=asyncio.Queue();self.sent=[];self.control=control;self.delegate=delegate
        self.ack=ack;self.stage=0;self.input=bytearray();self.output=np.arange(1920,dtype='<i2').tobytes()
        self.pending_original=pending_original
        self.provider_delay_s=provider_delay_s;self.ack_delay_s=ack_delay_s
    async def send(self,raw):
        e=json.loads(raw);self.sent.append(e)
        def put(e):self.queue.put_nowait(e)
        if e['type']=='session.start':put(dict(type='session.started',session=e['session']))
        if e['type']=='session.input_audio.append':
            data=base64.b64decode(e['audio']);self.input.extend(data);p=np.frombuffer(data,dtype='<i2')
            if self.stage==0 and (p==123).any():
                self.stage=1;put(dict(type='session.input_transcript.delta',delta='Actual original problem.',start_ms=0,end_ms=100))
                if self.delegate:put(dict(type='session.delegation.created',delegation=dict(id='original',target='client'),offset_ms=100))
                elif not self.control:put(dict(type='session.output_audio.delta',delta=base64.b64encode(self.output).decode()))
            if self.stage==1 and (p==456).any():
                self.stage=2;put(dict(type='session.input_transcript.delta',delta='Cancel it. What is two plus two?',start_ms=200,end_ms=300))
                put(dict(type='session.delegation.created',delegation=dict(id='cancel',target='client'),offset_ms=300))
        if e['type']=='session.commentary.append':
            if self.ack:
                async def acknowledge():
                    await asyncio.sleep(self.ack_delay_s/50)
                    put(dict(type='session.commentary.appended',client_event_id=e['event_id']))
                asyncio.create_task(acknowledge())
            put(dict(type='session.output_audio.delta',delta=base64.b64encode(self.output).decode()))
        if e['type']=='session.close':put(dict(type='session.closed',usage=dict(fixture_only=True)))
        await asyncio.sleep(0)
    def __aiter__(self):return self
    async def __anext__(self):return json.dumps(await self.queue.get())


class ReasoningStreamTests(unittest.IsolatedAsyncioTestCase):
    capture_impl=staticmethod(capture)
    async def run_capture(self,socket,clips):
        events=[];audio=bytearray();inputs=bytearray();calls=[];release=asyncio.Event()
        now=lambda:time.monotonic()*50
        async def sleep(seconds):await asyncio.sleep(seconds/50)
        async def runner(world,identity,model,out,env_file,*,started):
            payload=await world.start_backend(identity,model);calls.append(payload)
            out.mkdir();atomic_json(out/'transport.json',dict(http_call_started_monotonic=now(),actual_provider_transport=False))
            started.set()
            if socket.control and socket.pending_original and identity=='original':await release.wait()
            await sleep(socket.provider_delay_s)
            operation='cancel_reasoning' if identity=='cancel' else 'respond_to_user'
            args=dict(content='Delivery cancelled.' if identity=='cancel' else 'Actual computed answer.')
            if identity=='cancel':args['task_ids']=['original']
            r=await world.complete_backend(identity,dict(status='completed',id='fixture-'+identity,model=model,
                output=[dict(type='function_call',name=operation,arguments=json.dumps(args))]),actual_transport=False)
            if identity=='cancel':release.set()
            return r
        with tempfile.TemporaryDirectory()as folder:
            report,packets,transcripts=await self.capture_impl(socket,clips,Path(folder),events.append,
                backend_model='fixture-only',env_file=None,minimum_delay_s=0.,input_sink=inputs.extend,
                audio_sink=audio.extend,runner=runner,now=now,sleep=sleep)
            ledger=json.loads((Path(folder)/'lifecycle.json').read_text())
        self.assertEqual(bytes(inputs),bytes(socket.input));self.assertEqual(len(audio),sum(p['bytes'] for p in packets))
        return report,events,ledger,calls,bytes(inputs)

    async def test_cancel_runs_before_original_returns_and_followup_follows_actual_HTTP(self):
        a=np.full(3840,123,dtype='<i2');b=np.full(3840,456,dtype='<i2');socket=Socket(control=True)
        report,events,ledger,calls,raw=await self.run_capture(socket,[a,b])
        self.assertTrue(report['transport_complete'],report['errors']);self.assertEqual(len(calls),2)
        self.assertEqual(ledger['tasks']['original']['phase'],'stale_completion_preserved')
        self.assertFalse(ledger['tasks']['original']['actual_transport'])
        self.assertNotIn('Cancel it',calls[0]['input']);self.assertIn('Actual original problem',calls[1]['input'])
        self.assertIn('Cancel it',calls[1]['input'])
        e=next(e for e in events if e['type']=='local.followup.scheduled')
        self.assertGreaterEqual(e['first_sample']/24000,e['actual_first_HTTP_dispatch_at_s']+.56)
        sent=[e for e in socket.sent if e['type']=='session.commentary.append']
        self.assertEqual([e['delegation_id'] for e in sent],['cancel'])
        for pcm,start in zip((a,b),report['source_starts']):self.assertEqual(raw[start*2:(start+len(pcm))*2],pcm.tobytes())
        self.assertFalse(report['training_approved']);self.assertFalse(report['complete_native_episode_bound_verified'])

    async def test_direct_answer_has_no_forced_backend_and_preserves_original_input(self):
        a=np.full(3840,123,dtype='<i2');report,events,ledger,calls,raw=await self.run_capture(Socket(delegate=False),[a])
        self.assertTrue(report['transport_complete'],report['errors']);self.assertEqual(calls,[])
        self.assertEqual(ledger['tasks'],{});self.assertEqual(report['delegations'],0)
        start=report['source_starts'][0];self.assertEqual(raw[start*2:(start+len(a))*2],a.tobytes())

    async def test_cancelled_ready_result_does_not_hold_session_for_its_old_delay(self):
        report,events,ledger,calls,raw=await self.run_capture(Socket(control=True,pending_original=False),
            [np.full(3840,123,dtype='<i2'),np.full(3840,456,dtype='<i2')])
        self.assertTrue(report['transport_complete'],report['errors'])
        self.assertTrue(ledger['tasks']['original']['cancelled'])
        old=report['backend_HTTP_starts']['original']+report['first_result_minimum_delivery_delay_s']+RESPONSE_WINDOW_S
        self.assertLess(report['observed_until_s'],old)
        self.assertEqual([e['delegation_id'] for e in events if e['type']=='local.commentary.write'],['cancel'])

    async def test_missing_native_request_never_invents_dispatch_or_followup(self):
        report,events,ledger,calls,raw=await self.run_capture(Socket(control=True,delegate=False),
            [np.full(3840,123,dtype='<i2'),np.full(3840,456,dtype='<i2')])
        self.assertFalse(report['transport_complete']);self.assertEqual(calls,[])
        self.assertIsNone(report['source_starts'][1]);self.assertFalse((np.frombuffer(raw,dtype='<i2')==456).any())
        self.assertTrue(any(e['type']=='missing_native_dispatch_followup_not_invented' for e in report['errors']))

    async def test_missing_commentary_ack_is_failure_even_with_spoken_audio(self):
        report,*_=await self.run_capture(Socket(ack=False),[np.full(3840,123,dtype='<i2')])
        self.assertFalse(report['transport_complete']);self.assertGreater(report['output_packets'],0)
        self.assertTrue(any(e['type']=='missing_commentary_ack' for e in report['errors']))


if __name__=='__main__':unittest.main()
