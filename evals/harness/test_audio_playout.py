import unittest
import numpy as np
from evals.audio_playout import render_packets


class Playout(unittest.TestCase):
    def test_short_jitter_cannot_split_pcm_with_startup_buffer(self):
        packets = [(.1, np.ones(80)), (.185, np.ones(80)*2), (.261, np.ones(80)*3)]
        unbuffered, old = render_packets(packets, 1000, 0)
        buffered, new = render_packets(packets, 1000, .16)
        self.assertGreater(old["underrun_count"], 0)
        self.assertEqual(new["underrun_count"], 0)
        self.assertTrue(np.array_equal(buffered[260:], np.concatenate([p for _, p in packets])))
        for row in new["packets"]:
            self.assertGreaterEqual(row["playback_start_sample"] / 1000, row["available_s"])

    def test_real_stall_remains_visible_and_all_samples_remain(self):
        packets = [(.1, np.ones(80)), (.8, np.ones(80)*2)]
        output, report = render_packets(packets, 1000, .16)
        self.assertEqual(report["underrun_count"], 1)
        self.assertEqual(report["underrun_samples"], 460)
        self.assertEqual(report["source_samples"], 160)
        self.assertTrue(np.array_equal(output[800:], packets[1][1]))

    def test_invalid_packet_order_is_not_silently_retimed(self):
        with self.assertRaises(ValueError):
            render_packets([(.2, np.ones(10)), (.1, np.ones(10))])


if __name__ == "__main__":
    unittest.main()
