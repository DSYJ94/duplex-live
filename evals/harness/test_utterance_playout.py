import unittest
import numpy as np
from evals.utterance_playout import render_utterances


class UtterancePlayoutTests(unittest.TestCase):
    def test_each_new_utterance_needs_the_same_startup_buffer(self):
        packets=[{'event_id':name,'available_s':at,'pcm':np.ones(length)} for name,at,length in
            [('one',0.,160),('one',.32,320),('two',2.,160),('two',2.32,320)]]
        output,report=render_utterances(packets,1000,.16)
        self.assertEqual(report['within_utterance_underrun_samples'],0)
        self.assertTrue(np.array_equal(output[2160:2640],np.ones(480)))

    def test_interrupt_cuts_queued_old_pcm_but_keeps_an_unrelated_event(self):
        packets=[{'event_id':'old','available_s':0.,'pcm':np.ones(1000)},
            {'event_id':'new','available_s':.7,'pcm':np.ones(100)*2}]
        output,report=render_utterances(packets,1000,.16,{'old':.5})
        self.assertEqual(report['cancelled_before_playback_samples'],660)
        self.assertTrue(np.array_equal(output[160:500],np.ones(340)))
        self.assertFalse(np.any(output[500:860]))
        self.assertTrue(np.array_equal(output[860:960],np.ones(100)*2))
        self.assertTrue(report['sample_accounting_exact'])

    def test_late_packet_remains_cancelled_and_never_restarts_the_voice(self):
        packets=[{'event_id':'old','available_s':.6,'pcm':np.ones(100)}]
        output,report=render_utterances(packets,1000,.16,{'old':.5})
        self.assertEqual(len(output),0)
        self.assertEqual(report['cancelled_before_playback_samples'],100)

    def test_real_stall_is_visible_and_overlap_is_not_silently_mixed(self):
        packets=[{'event_id':'one','available_s':0.,'pcm':np.ones(160)},
            {'event_id':'one','available_s':.8,'pcm':np.ones(100)}]
        _,report=render_utterances(packets,1000,.16)
        self.assertEqual(report['within_utterance_underrun_samples'],480)
        with self.assertRaises(ValueError):
            render_utterances([{'event_id':'one','available_s':0.,'pcm':np.ones(1000)},
                {'event_id':'two','available_s':.3,'pcm':np.ones(100)}],1000,.16)


if __name__=='__main__':
    unittest.main()
