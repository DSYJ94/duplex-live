"""Recognition windows must preserve voiced samples and the original clock."""
import unittest

import numpy as np

from evals.moshi_speech_audit import asr_window, restore_clock


class ASRWindowTests(unittest.TestCase):
    def test_leading_quiet_is_removed_without_losing_signal(self):
        pcm=np.zeros(16000*5,dtype=np.float32)
        pcm[32000:40000]=.1
        audio,head,tail,silent=asr_window(pcm,True,True,False)
        self.assertFalse(silent)
        self.assertEqual(head,32000-4000)
        self.assertEqual(tail,len(pcm)-(40000+4000))
        np.testing.assert_array_equal(audio,pcm[head:len(pcm)-tail])
        self.assertEqual(np.count_nonzero(audio),np.count_nonzero(pcm))
        result={'text':'Hello.','segments':[{'start':.25,'end':.75,
                'words':[{'text':'Hello.','start':.25,'end':.75}]}]}
        shifted=restore_clock(result,head/16000)
        self.assertEqual(shifted['segments'][0]['words'][0]['start'],2.)
        self.assertEqual(shifted['segments'][0]['words'][0]['end'],2.5)
        self.assertEqual(shifted['text'],result['text'])
        self.assertEqual(result['segments'][0]['start'],.25)

    def test_default_does_not_trim_leading_quiet_or_change_clock(self):
        pcm=np.r_[np.zeros(32000),np.full(8000,.1)].astype(np.float32)
        audio,head,tail,silent=asr_window(pcm,False,False,False)
        np.testing.assert_array_equal(audio,pcm)
        self.assertEqual((head,tail,silent),(0,0,False))

    def test_no_internal_gap_is_removed(self):
        pcm=np.zeros(16000*6,dtype=np.float32)
        pcm[16000:20000]=.1;pcm[64000:68000]=.1
        audio,head,tail,_=asr_window(pcm,True,True,False)
        np.testing.assert_array_equal(audio,pcm[head:len(pcm)-tail])
        self.assertEqual(np.count_nonzero(audio[20000-head:64000-head]),0)

    def test_silent_input_remains_explicit_failure(self):
        pcm=np.zeros(16000,dtype=np.float32)
        _,head,tail,silent=asr_window(pcm,True,True,True)
        self.assertTrue(silent)
        self.assertEqual((head,tail),(0,0))
        with self.assertRaises(RuntimeError):asr_window(pcm,True,True,False)


if __name__=='__main__':unittest.main()
