"""Reproduce premature close during actual output; only recorder closure changes."""
import base64
import json
import time
import unittest
import numpy as np

from evals.harness import test_native_reasoning_stream as original
from evals.harness import test_native_reasoning_stream_v3 as causal
from evals import native_reasoning_stream_v5 as previous
from evals import native_reasoning_stream_v6 as revised


class SpeakingSocket(original.Socket):
    def __init__(self):
        super().__init__(); self.result_at=None; self.final_word_sent=False

    async def send(self, raw):
        event=json.loads(raw)
        if event['type']=='session.commentary.append':self.result_at=time.monotonic()*50
        await super().send(raw)
        if event['type']=='session.input_audio.append' and self.result_at is not None:
            elapsed=time.monotonic()*50-self.result_at
            if 4. <= elapsed < 7.:
                self.queue.put_nowait(dict(type='session.output_audio.delta',
                    delta=base64.b64encode(self.output).decode()))
            if elapsed>=6.5 and not self.final_word_sent:
                self.final_word_sent=True
                self.queue.put_nowait(dict(type='session.output_transcript.delta',delta='total points twenty-nine.'))


class CompleteSpeech(causal.CausalStream):
    capture_impl=staticmethod(revised.capture)

    async def test_finishes_speech_before_close_and_retains_all_input_audio(self):
        socket=SpeakingSocket(); report,events,_,_,raw=await self.run_capture(socket,[np.full(3840,123,dtype='<i2')])
        self.assertTrue(report['transport_complete'],report['errors'])
        self.assertTrue(socket.final_word_sent)
        close=next(e for e in events if e['type']=='local.session.close.requested')
        self.assertGreaterEqual(close['received_at_s'],close['output_quiet_deadline_s'])
        self.assertEqual(close['required_output_quiet_s'],1.)
        self.assertEqual(len(raw)//2,report['sent_input_samples'])
        self.assertEqual(revised.configuration(),previous.configuration())
        self.assertEqual(revised.MAXIMUM_INPUT_S,previous.MAXIMUM_INPUT_S)
        self.assertFalse(report['native_request_forced'])


class OriginalCutoff(unittest.IsolatedAsyncioTestCase):
    capture_impl=staticmethod(previous.capture)
    run_capture=original.ReasoningStreamTests.run_capture

    async def test_previous_fixed_window_closes_before_final_words(self):
        socket=SpeakingSocket(); report,*_=await self.run_capture(socket,[np.full(3840,123,dtype='<i2')])
        self.assertTrue(report['transport_complete'])
        self.assertFalse(socket.final_word_sent)


class OriginalSamples(unittest.TestCase):
    def test_active_playout_end_uses_original_pcm_without_gain(self):
        pcm=np.zeros(1920,dtype='<i2');pcm[480:960]=1000;before=pcm.copy()
        self.assertEqual(revised.last_active_sample(pcm.tobytes(),10000),10960)
        self.assertIsNone(revised.last_active_sample(np.zeros(1920,dtype='<i2').tobytes(),10000))
        np.testing.assert_array_equal(pcm,before)


if __name__=='__main__':unittest.main()
