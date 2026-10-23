import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from evals import native_reasoning_source_quality_v4 as quality
from evals.native_reasoning_source import sha


class CompletePacingQualityTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';source.mkdir();asr=root/'asr';asr.mkdir()
        identities=[quality.OBJECT]+[f'{i:064x}'for i in range(39)]
        objects={i:dict(input='Two badges.')for i in identities};scenes=[dict(id=i)for i in range(32)]
        pcm=np.full(24000,.1,dtype=np.float32)
        result=dict(text='Two badges.',segments=[dict(words=[dict(text='Two',start=.05,end=.3),dict(text=' badges.',start=.35,end=.7)])])
        rows=[];old=[]
        for identity in identities:
            wav=source/(identity+'.wav');sf.write(wav,pcm,24000,subtype='PCM_16')
            (asr/(identity+'-asr.json')).write_text(json.dumps(result))
            loaded,rate=sf.read(wav,dtype='float32')
            rows.append(dict(object_id=identity,wav=wav.name))
            old.append(dict(object_id=identity,wav_sha256=sha(wav),asr_sha256=sha(asr/(identity+'-asr.json')),
                            **quality.previous.assess('Two badges.',result,loaded,rate)))
        return source,asr,dict(objects=rows,scenes=scenes),dict(results=old),(objects,scenes)

    def test_every_object_is_reassessed_and_only_changed_object_uses_new_recognition(self):
        with tempfile.TemporaryDirectory()as folder:
            source,asr,manifest,parent,binding=self.fixture(Path(folder))
            with patch.object(quality,'requests_for',return_value=binding):r=quality.rows(source,manifest,{},asr,parent)
            self.assertEqual(len(r),40);self.assertEqual(sum(v['independent_asr_reused']for v in r),39)
            self.assertTrue(all(v['passed']for v in r))

    def test_replacing_the_changed_transcript_with_wrong_words_fails_the_whole_set(self):
        with tempfile.TemporaryDirectory()as folder:
            source,asr,manifest,parent,binding=self.fixture(Path(folder))
            p=asr/(quality.OBJECT+'-asr.json');r=json.loads(p.read_text());r['text']='Three candles.';p.write_text(json.dumps(r))
            with patch.object(quality,'requests_for',return_value=binding):v=quality.rows(source,manifest,{},asr,parent)
            self.assertEqual(len(v),40);self.assertFalse(all(x['passed']for x in v));self.assertFalse(v[0]['passed'])

    def test_changed_reused_waveform_or_ASR_cannot_inherit_old_approval(self):
        for mutation in ('wave','asr'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as folder:
                source,asr,manifest,parent,binding=self.fixture(Path(folder));identity=manifest['objects'][1]['object_id']
                if mutation=='wave':sf.write(source/(identity+'.wav'),np.full(24000,.2),24000)
                else:
                    p=asr/(identity+'-asr.json');p.write_text(p.read_text()+'\n')
                with patch.object(quality,'requests_for',return_value=binding),self.assertRaises(ValueError):
                    quality.rows(source,manifest,{},asr,parent)

    def test_missing_duplicate_or_reordered_source_membership_rejects(self):
        for mutation in ('missing','duplicate','order'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory()as folder:
                source,asr,manifest,parent,binding=self.fixture(Path(folder));manifest=copy.deepcopy(manifest)
                if mutation=='missing':manifest['objects'].pop()
                elif mutation=='duplicate':manifest['objects'][-1]=manifest['objects'][0]
                else:manifest['objects'].reverse()
                with patch.object(quality,'requests_for',return_value=binding),self.assertRaises(ValueError):
                    quality.rows(source,manifest,{},asr,parent)


if __name__=='__main__':unittest.main()
