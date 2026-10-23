import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from evals import native_reasoning_teacher_mixed_speech as speech
from evals.native_reasoning_source import sha


class MixedSpeechTests(unittest.TestCase):
    def fixture(self,root):
        batch=root/'batch';batch.mkdir();worker=root/'worker';worker.mkdir();records=[];recognized=[]
        raw=dict(text='Eight tokens.',segments=[dict(words=[dict(text='Eight',start=.1,end=.2),
                 dict(text='tokens.',start=.2,end=.4)])],speech_activity=[dict(start=0.,end=.5)])
        for identity in speech.mixed.IDS:
            folder=batch/identity;folder.mkdir();wav=folder/'model_out.wav'
            sf.write(wav,np.full(12000,.1,dtype=np.float32),24000,subtype='PCM_16')
            (folder/'transcripts.json').write_text(json.dumps([dict(type='session.output_transcript.delta',delta='Eight tokens.')]))
            (folder/'report.json').write_text(json.dumps(dict(native_frame_budget=dict(frame_count_diagnostic_pass=True))))
            name=identity+'-asr.json';(worker/name).write_text(json.dumps(raw))
            recognized.append(dict(id=identity,asr_file=name,wav_sha256=sha(wav),asr_sha256=sha(worker/name)))
            records.append(dict(id=identity,recorder='v6',recording=dict(artifacts={'model_out.wav':{'sha256':sha(wav)}})))
        (worker/'recognition.json').write_text(json.dumps(dict(recordings=recognized)))
        header=dict(complete_cases=32,recordings=records);(batch/'teacher-collection.json').write_text(json.dumps(header))
        return SimpleNamespace(region_worker=worker),batch,header

    def test_every_original_case_is_reassessed_and_partial_table_rejected(self):
        with tempfile.TemporaryDirectory()as folder:
            c,batch,header=self.fixture(Path(folder));rows=speech.rows(c,batch,header)
            self.assertEqual(len(rows),32);self.assertTrue(all(r['passed']for r in rows))
            header['recordings'].pop()
            with self.assertRaisesRegex(ValueError,'All32'):speech.rows(c,batch,header)

    def test_changed_reused_waveform_or_independent_asr_rejects(self):
        for target in ('wave','asr'):
            with self.subTest(target=target),tempfile.TemporaryDirectory()as folder:
                c,batch,header=self.fixture(Path(folder));identity=speech.mixed.IDS[-1]
                path=(batch/identity/'model_out.wav')if target=='wave'else(c.region_worker/(identity+'-asr.json'))
                path.write_bytes(path.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError,'byte binding'):speech.rows(c,batch,header)

    def test_one_failed_word_case_keeps_whole32_failed_and_subset_cannot_approve(self):
        with tempfile.TemporaryDirectory()as folder:
            c,batch,header=self.fixture(Path(folder));rows=speech.rows(c,batch,header)
            rows[-1]['passed']=False;rows[-1]['reasons']=['Independent exact-word agreement failed']
            with patch.object(speech.mixed,'source_pins',return_value={'fixture':'unchanged'}):
                value=dict(speech.body(c,batch,rows),at_utc='fixture')
                self.assertFalse(value['passed']);speech.verify_value(c,batch,value,rows,False)
                with self.assertRaises(ValueError):speech.verify_value(c,batch,value,rows)
                subset=copy.deepcopy(value);subset['results']=subset['results'][:-1];subset['passed']=True
                with self.assertRaises(ValueError):speech.verify_value(c,batch,subset,rows,False)


if __name__=='__main__':unittest.main()
