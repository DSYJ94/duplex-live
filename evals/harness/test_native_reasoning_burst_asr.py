import unittest
from evals.native_reasoning_burst_asr import restore


class ClockRestoration(unittest.TestCase):
    def test_separate_utterances_keep_the_original_gap_and_word_identity(self):
        raw=dict(text='badge',segments=[dict(id=0,start=.2,end=.8,
            words=[dict(text='badge',start=.3,end=.7)])])
        a=restore(raw,24000,48000,24000,0)
        b=restore(raw,240000,264000,24000,1)
        self.assertEqual(a['segments'][0]['words'][0]['start'],1.3)
        self.assertEqual(b['segments'][0]['words'][0]['start'],10.3)
        self.assertEqual(b['segments'][0]['id'],1)
        self.assertEqual(raw['segments'][0]['words'][0]['start'],.3)

    def test_clock_outside_region_cannot_be_relabelled(self):
        with self.assertRaises(ValueError):
            restore(dict(segments=[dict(start=0.,end=2.)]),0,24000,24000,0)


if __name__ == '__main__':
    unittest.main()
