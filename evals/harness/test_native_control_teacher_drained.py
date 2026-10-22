"""Real recorder seam with a delayed output backlog; fixture evidence only."""
import asyncio
import base64
import json
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from evals import native_control_teacher_drained as drained
from evals.audio_playout import render_packets
from evals.harness.test_native_control_teacher import Clock, Socket

class BacklogSocket(Socket):
    def __init__(self):
        super().__init__()
        self.output = np.arange(6*24000, dtype='int16').tobytes()
    async def send(self, raw):
        message=json.loads(raw)
        self.sent.append(message)
        if message['type']=='session.start':
            self.queue.put_nowait(dict(type='session.started',session=message['session']))
        elif message['type']=='session.input_audio.append':
            self.input_count+=1
            if self.input_count==189:
                self.queue.put_nowait(dict(type='session.output_audio.delta',delta=base64.b64encode(self.output).decode()))
        elif message['type']=='session.close':
            self.queue.put_nowait(dict(type='session.closed',usage=dict(fixture_only=True)))
        await asyncio.sleep(0)

class PlayoutDrain(unittest.IsolatedAsyncioTestCase):
    async def test_capture_waits_for_real_backlog_without_changing_any_input_or_output(self):
        clock,socket,events=Clock(),BacklogSocket(),[]
        samples=np.zeros(1920+15*24000,dtype='int16')
        report,audio,packets,_=await drained.capture_stream(socket,samples,1920,events.append,
                                                         now=clock.now,sleep=clock.sleep)
        self.assertTrue(report['session_finalized'])
        self.assertEqual(report['errors'],[])
        self.assertEqual(audio,socket.output)
        sent=b''.join(base64.b64decode(e['audio']) for e in socket.sent if e['type']=='session.input_audio.append')
        self.assertEqual(sent,samples.tobytes())
        played,_=render_packets([(p['arrival_s'],np.frombuffer(audio[p['byte_offset']:p['byte_offset']+p['bytes']],dtype='<i2')) for p in packets],24000,0)
        self.assertLessEqual(len(played)/24000,report['observed_until_s']+.08)
        self.assertLess(report['server_closed_at_s'],len(played)/24000)
        self.assertEqual(report['server_closed_at_s'],next(e['received_at_s'] for e in events if e['type']=='session.closed'))
        self.assertFalse(report['training_approved'])
        self.assertTrue(drained.verify_drain(events,report)['actual_postclose_drain_verified'])
        broken=[dict(e) for e in events]
        end=next(e for e in broken if e['type']=='local.output.drain.completed')
        end['received_at_s']-=4
        with self.assertRaisesRegex(ValueError,'observation ended'):
            drained.verify_drain(broken,report)

    async def test_capture_one_and_verifier_preserve_full_raw_record_and_reject_drain_tampering(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sid=drained.FAILED_IDS[0]; (root/sid).mkdir()
            sf.write(root/sid/'stem_user.wav',np.full(1920,.1,dtype='float32'),24000,subtype='FLOAT')
            (root/'raw-source.json').write_text('{"fixture_only":true}')
            clock,socket=Clock(),BacklogSocket()
            class Connection:
                async def __aenter__(self): return socket
                async def __aexit__(self,*args): return None
            capture=drained.capture_stream
            async def run(*args,**kwargs): return await capture(*args,**kwargs,now=clock.now,sleep=clock.sleep)
            folder=root/'record'
            with patch.object(drained,'capture_stream',side_effect=run):
                record=await drained.capture_one(root,dict(id=sid,frames=1920),folder,'fixture-audit',
                                                'fixture-key',connector=lambda *a,**kw:Connection())
            drained.verify_exact_record(root,dict(id=sid,frames=1920),folder,'fixture-audit',record)
            events=[json.loads(e) for e in (folder/'events.jsonl').read_text().splitlines()]
            next(e for e in events if e['type']=='local.output.drain.completed')['playout_end_sample']-=1
            (folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            record['artifacts']['events.jsonl']=drained.sha256(folder/'events.jsonl')
            with self.assertRaisesRegex(ValueError,'observation ended'):
                drained.verify_exact_record(root,dict(id=sid,frames=1920),folder,'fixture-audit',record)

    async def test_only_one_new_session_is_allowed_before_accessing_data(self):
        for n in (0,2,True):
            with self.assertRaisesRegex(ValueError,'Exactly one'):
                await drained.collect(*(Path('absent') for _ in range(6)),'fixture-key',Path('lease'),concurrency=n)

    async def test_excessive_backlog_fails_closed_without_trimming(self):
        clock,socket=Clock(),BacklogSocket()
        async def trigger(ws,pcm,source_samples,emit,**kwargs):
            kwargs['now']()
            emit(dict(type='session.output_audio.delta',received_at_s=0.,audio_bytes=30*48000))
            return dict(session_finalized=True,errors=[],observed_until_s=0.),b'raw',[],[]
        with patch.object(drained.original,'capture_stream',side_effect=trigger):
            report,audio,_,_=await drained.capture_stream(socket,np.zeros(1920,dtype='int16'),1920,lambda e:None,
                                                       now=clock.now,sleep=clock.sleep)
        self.assertEqual(report['errors'],[dict(error_type='TimeoutError',component='output_drain')])
        self.assertEqual(audio,b'raw')

    def test_new_quality_schema_cannot_bypass_whole40_approval(self):
        from evals import native_control_teacher_drained_quality as quality
        from evals import native_control_semantic as semantic
        from evals import native_control_postteacher_pipeline as pipeline
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); q=root/'quality.json';q.write_text(json.dumps(dict(schema=quality.SCHEMA)))
            with patch.object(quality,'verify',side_effect=ValueError('Whole40 failed')) as verify:
                with self.assertRaisesRegex(ValueError,'Whole40 failed'):semantic.verify_quality(root,root,root,q)
                verify.assert_called_once_with(root,root,root,q)
        rows=list(pipeline.stages(*(Path(x) for x in ('raw','source','batch','parent','quality','out','weights','env')),drained_teacher=True))
        self.assertEqual(rows[0][1],'evals.native_control_teacher_drained_quality')
        self.assertEqual(rows[1][1],'evals.native_control_teacher_drained_quality')
        self.assertIn('--verify',rows[1][2])
        self.assertEqual(rows[2][1],'evals.native_control_semantic')
        with self.assertRaises(ValueError):list(pipeline.stages(*(Path('fixture') for _ in range(8)),drained_teacher=True,exact_teacher=True))

if __name__=='__main__': unittest.main()
