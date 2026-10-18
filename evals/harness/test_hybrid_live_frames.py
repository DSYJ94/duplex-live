import struct
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from serving.hybrid.frames import LiveFrames
from serving.hybrid.server import Output


def packet(sequence,value=0.):return struct.pack('<I',sequence)+np.full(1920,value,dtype='<f4').tobytes()


class LiveCaptureProtocol(unittest.TestCase):
    def test_aborted_capture_exits_media_actor_without_draining_synthetic_tail(self):
        from evals.duplex_hybrid_probe import run_case
        source=LiveFrames();source.abort()
        listener=SimpleNamespace(spm=SimpleNamespace(decode=lambda ids:''),reset=lambda:None)
        speaker=SimpleNamespace(stop=lambda:None)
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'capture'
            with self.assertRaises(ConnectionAbortedError):
                run_case(listener,speaker,dict(split='live-transport',scenario_id='fixture',world={}),out,0,worker_kind='reasoning-development',live_input=source)
            self.assertTrue((out/'capture-arrivals.json').exists())
            self.assertFalse((out/'model.wav').exists())

    def test_ordered_actual_pcm_is_preserved_and_ended_capture_drains(self):
        source=LiveFrames(2);source.feed(packet(0,.25));source.feed(packet(1,-.5));source.finish()
        self.assertFalse(source.finished);first,arrival=source.take();second,_=source.take();self.assertTrue(source.finished)
        np.testing.assert_array_equal(first,np.full(1920,.25,dtype=np.float32));np.testing.assert_array_equal(source.pcm(),np.concatenate([first,second]))
        self.assertGreater(arrival,0);self.assertEqual([r['sequence'] for r in source.receipts],[0,1])
        with self.assertRaises(ValueError):source.feed(packet(2))
    def test_backpressure_duplicate_malformed_nonfinite_and_out_of_range_reject(self):
        source=LiveFrames();source.feed(packet(0))
        for encoded in (packet(0),b'cut',packet(1,float('nan')),packet(1,9.)):
            with self.assertRaises(ValueError):source.feed(encoded)
        for n in range(1,16):source.feed(packet(n))
        with self.assertRaisesRegex(RuntimeError,'backlog'):source.feed(packet(16))
        self.assertEqual(len(source.received),16);self.assertEqual(source.sequence,16)
    def test_output_sequence_and_event_identity_preserve_pcm_and_disconnect_is_explicit(self):
        out=Output();pcm=np.linspace(-1,1,1920,dtype=np.float32);out.pcm('speech-4',pcm,1.)
        encoded=out.queue.get_nowait();self.assertEqual(struct.unpack_from('<II',encoded),(4,0));np.testing.assert_array_equal(np.frombuffer(encoded,dtype='<f4',offset=8),pcm)
        out.disconnected.set();out.pcm('speech-4',pcm,2.);self.assertTrue(out.queue.empty())


if __name__=='__main__':unittest.main()
