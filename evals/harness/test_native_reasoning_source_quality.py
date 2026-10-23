import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from evals.native_reasoning_source_quality import assess,persisted_asr


class ReasoningSourceQuality(unittest.TestCase):
    def setUp(self):
        self.pcm=np.full(24000,.1,dtype=np.float32)
        self.result=dict(text='Two badges.',segments=[dict(words=[dict(text='Two',start=.05,end=.3),dict(text=' badges.',start=.35,end=.7)])])
    def test_independent_words_and_audio_pass_without_reference_prompt(self):
        v=assess('Two badges.',self.result,self.pcm,24000)
        self.assertTrue(v['passed']);self.assertEqual(v['raw_wer'],0.);self.assertFalse(v['reference_text_prompt_used'])
    def test_mismatch_clipping_and_invented_intervals_reject(self):
        for mutate in (lambda x:x.update(text='Three candles.'),
                       lambda x:x['segments'][0]['words'][0].update(start=-1),
                       lambda x:x['segments'][0]['words'][1].update(end=2),
                       lambda x:x['segments'][0]['words'][1].update(text='wrong')):
            r=copy.deepcopy(self.result);mutate(r)
            self.assertFalse(assess('Two badges.',r,self.pcm,24000)['passed'])
        self.assertFalse(assess('Two badges.',self.result,np.ones(24000),24000)['passed'])
    def test_new_long_question_schema_keeps_native_full_episode_bound_and_old_limits_unchanged(self):
        self.assertTrue(assess('Two badges.',self.result,np.full(35*24000,.1),24000)['passed'])
        self.assertFalse(assess('Two badges.',self.result,np.full(89*24000,.1),24000)['passed'])
        self.assertFalse(assess('Two badges.',self.result,self.pcm,16000)['passed'])

    def test_existing_numeral_expansion_is_preserved_and_wrong_values_still_fail(self):
        r=copy.deepcopy(self.result);r['text']='2 badges.';r['segments'][0]['words'][0]['text']='2'
        self.assertTrue(assess('Two badges.',r,self.pcm,24000)['passed'])
        r['text']='3 badges.';r['segments'][0]['words'][0]['text']='3'
        self.assertFalse(assess('Two badges.',r,self.pcm,24000)['passed'])

    def test_numpy_clocks_are_assessed_from_the_same_persisted_values_as_verification(self):
        r=copy.deepcopy(self.result)
        for word in r['segments'][0]['words']:
            word['start']=np.float64(word['start']);word['end']=np.float64(word['end'])
        self.assertFalse(assess('Two badges.',r,self.pcm,24000)['passed'])
        with tempfile.TemporaryDirectory()as folder:
            path=Path(folder)/'asr.json';saved=persisted_asr(path,r)
            self.assertEqual(saved,json.loads(path.read_text()))
            self.assertEqual(saved,r)
            self.assertTrue(assess('Two badges.',saved,self.pcm,24000)['passed'])

    def test_readback_cannot_hide_invalid_or_nonfinite_clock_values(self):
        for bad in [-1.,float('nan'),float('inf'),2.]:
            r=copy.deepcopy(self.result);r['segments'][0]['words'][0]['end']=np.float64(bad)
            with tempfile.TemporaryDirectory()as folder:
                saved=persisted_asr(Path(folder)/'asr.json',r)
                self.assertFalse(assess('Two badges.',saved,self.pcm,24000)['passed'])


if __name__=='__main__':unittest.main()
