"""Real codec and lifecycle checks; these do not approve any teacher recording."""
import asyncio
from fractions import Fraction
import json
import unittest
import numpy as np
from av import AudioFrame, AudioResampler
from aiortc.codecs.opus import OpusEncoder, OpusDecoder
from aiortc.jitterbuffer import JitterFrame
from evals import native_control_webrtc_probe as probe
from evals.native_control_teacher_scoped import configuration


class WebRTCProbe(unittest.TestCase):
    def test_only_wire_format_is_changed_from_scoped_reference(self):
        expected = configuration()
        expected['audio'].pop('format')
        self.assertEqual(probe.configuration(), expected)
        raw = {'type':'session.started', 'session':{'sdp':'secret', 'api_key':'secret',
               'nested':[{'credential':'secret'}], 'model':'gpt-live-1'}}
        cleaned = json.dumps(probe.redact(raw))
        self.assertNotIn('secret', cleaned)
        self.assertIn('gpt-live-1', cleaned)

    def test_actual_opus_roundtrip_and_explicit_duplicate_mono_conversion(self):
        encoder, decoder = OpusEncoder(), OpusDecoder()
        resampler = AudioResampler(format='s16', layout='mono', rate=24000)
        decoded = []
        for i in range(12):
            t = np.arange(i*480, (i+1)*480)
            samples = np.round(6000*np.sin(2*np.pi*440*t/24000)).astype(np.int16)
            frame = AudioFrame.from_ndarray(samples[None], format='s16', layout='mono')
            frame.sample_rate, frame.pts, frame.time_base = 24000, i*480, Fraction(1,24000)
            payloads, timestamp = encoder.encode(frame)
            for payload in payloads:
                for remote in decoder.decode(JitterFrame(payload, timestamp)):
                    plane = probe.mono_plane(remote)
                    self.assertEqual(plane.shape, (1, remote.samples))
                    mono = AudioFrame.from_ndarray(plane, format='s16', layout='mono')
                    mono.sample_rate, mono.pts, mono.time_base = remote.sample_rate, remote.pts, remote.time_base
                    decoded.extend(x.to_ndarray() for x in resampler.resample(mono))
        decoded.extend(x.to_ndarray() for x in resampler.resample(None))
        pcm = np.concatenate(decoded, axis=1)
        self.assertGreater(pcm.size, 4000)
        self.assertGreater(np.max(np.abs(pcm)), 1000)
        self.assertLess(np.max(np.abs(pcm)), 10000)
        different = np.array([[1,2,3,4]], dtype=np.int16)
        stereo = AudioFrame.from_ndarray(different, format='s16', layout='stereo')
        with self.assertRaisesRegex(ValueError, 'channels differ'):
            probe.mono_plane(stereo)

    def test_real_observation_waits_for_every_original_sample(self):
        now = [10.]
        events = []
        async def sleep(delay): now[0] += delay
        rows = [{'arrival_s':9., 'bytes':96000}, {'arrival_s':9.1, 'bytes':48000}]
        ended = asyncio.run(probe.observe_playout(rows, 0., events.append, now=lambda:now[0], sleep=sleep))
        self.assertAlmostEqual(ended, 12.16)
        self.assertEqual(events[-1]['playout_end_sample'], 288000)
        with self.assertRaises(TimeoutError):
            asyncio.run(probe.observe_playout([{'arrival_s':40., 'bytes':48000}], 0., events.append,
                        now=lambda:now[0], sleep=sleep))

    def test_failure_cleanup_closes_same_session_without_restarting(self):
        sent, urls, events = [], [], []
        class Socket:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def send(self, raw): sent.append(json.loads(raw))
            def __aiter__(self):
                async def values():
                    yield json.dumps({'type':'session.closed', 'session':{'client_secret':'sensitive'}})
                return values()
        def connect(url, **kwargs): urls.append(url); return Socket()
        result = asyncio.run(probe.close_sideband('live_fixture', 'not-a-real-key', events.append, connector=connect))
        self.assertTrue(result)
        self.assertEqual(sent, [{'type':'session.close'}])
        self.assertEqual(urls, ['wss://api.openai.com/v1/live/sessions/live_fixture/attach'])
        self.assertNotIn('sensitive', json.dumps(events))


if __name__ == '__main__': unittest.main()
