"""Artifact and whole-sequence tests; no actual model or provider calls."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf

from evals import native_reasoning_teacher as teacher
from evals.native_reasoning_stream import capture
from evals.harness.test_native_reasoning_stream import Socket


class TeacherArtifacts(unittest.IsolatedAsyncioTestCase):
    implementation=teacher
    capture_stream=staticmethod(capture)
    async def make_record(self,root,*,broken_exit=False):
        source=root/'source';source.mkdir();wav=source/'user.wav';pcm=np.full(3840,123,dtype='int16')
        sf.write(wav,pcm,24000,subtype='PCM_16');(source/'source-audio.json').write_text('{}')
        manifest=dict(objects=[dict(object_id='one',wav='user.wav',wav_sha256=teacher.sha(wav))])
        scene=dict(id='fixture-only',clips=[dict(object_id='one')]);socket=Socket(delegate=False)
        class Connection:
            async def __aenter__(self):return socket
            async def __aexit__(self,*args):
                if broken_exit:raise ConnectionError('fixture finalization failure')
        async def wrapped(*args,**kwargs):
            return await self.capture_stream(*args,**kwargs,now=lambda:time.monotonic()*50,
                                 sleep=lambda delay:__import__('asyncio').sleep(delay/50))
        tokenizer=SimpleNamespace(encode=lambda text:text.split());folder=root/'record'
        row=dict(annotations_not_model_inputs=dict(minimum_result_delivery_delay_s=0.,oracle='must-not-enter-model'))
        with patch.object(self.implementation,'source_manifest',return_value=(manifest,{})),patch.object(self.implementation,'capture',side_effect=wrapped):
            record=await self.implementation.capture_one(source,scene,row,folder,'fixture-quality','fixture-only',None,tokenizer,
                                             connector=lambda *a,**kw:Connection())
        return source,scene,folder,record,pcm,socket,tokenizer

    async def test_original_input_and_every_output_sample_survive_without_fixture_approval(self):
        with tempfile.TemporaryDirectory()as directory:
            source,scene,folder,record,pcm,socket,tokenizer=await self.make_record(Path(directory))
            report=json.loads((folder/'report.json').read_text());self.assertTrue(report['transport_complete'])
            self.assertFalse(report['actual_provider_transport']);self.assertFalse(report['teacher_or_training_approved'])
            raw=(folder/'input-stream.pcm').read_bytes();start=report['source_starts'][0]
            self.assertEqual(raw[start*2:(start+len(pcm))*2],pcm.astype('<i2').tobytes())
            self.assertEqual((folder/'output-stream.pcm').read_bytes(),socket.output)
            saved,rate=sf.read(folder/'packet-stream.wav',dtype='int16');self.assertEqual(saved.tobytes(),socket.output)
            env=json.loads((folder/'environment.json').read_text());self.assertNotIn('must-not-enter-model',json.dumps(env['session_config']))
            with self.assertRaisesRegex(ValueError,'actual unchanged'):self.implementation.verify_record(source,scene,folder,record,tokenizer)

    async def test_context_manager_failure_keeps_spools_and_rendered_output(self):
        with tempfile.TemporaryDirectory()as directory:
            _,_,folder,record,_,socket,_=await self.make_record(Path(directory),broken_exit=True)
            report=json.loads((folder/'report.json').read_text());self.assertFalse(record['transport_complete'])
            self.assertEqual((folder/'output-stream.pcm').read_bytes(),socket.output)
            self.assertTrue((folder/'model_out.wav').is_file())
            self.assertTrue(any(e['type']=='connection_context_failure' for e in report['errors']))


class FullFrameAccounting(unittest.TestCase):
    def test_delivered_then_cancelled_span_is_retained_and_overbound_is_rejected(self):
        tokenizer=SimpleNamespace(encode=lambda text:text.split())
        ledger=dict(tasks={'old':dict(cancelled=True,wire_write_completed_at_s=1.,receipt=dict(content='Old delivered content.')),
                           'new':dict(wire_write_completed_at_s=2.,receipt=dict(content='New delivered content.')),
                           'never-sent':dict(cancelled=True,receipt=dict(content='Not delivered.'))})
        budget=teacher.frame_budget(dict(observed_until_s=69.),ledger,tokenizer)
        self.assertEqual(budget['full_measured_prefix_live_and_span_frames'],232+863+10)
        self.assertFalse(budget['frame_count_diagnostic_pass']);self.assertEqual(len(budget['actual_sent_context_spans']),2)
        self.assertFalse(budget['original_audio_cropped']);self.assertFalse(budget['complete_native_assembly_and_numerical_approval'])


if __name__=='__main__':unittest.main()
