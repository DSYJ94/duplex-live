"""Received words must independently match the entire original user conversation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from evals import native_control_webrtc_quality as quality


class WebRTCQuality(unittest.TestCase):
    def test_independent_received_audio_needs_all_user_words_and_valid_times(self):
        scene=dict(turns=[dict(role='user',text='Stay quiet.'),dict(role='assistant',text='Unused'),
                          dict(role='user',text='Until marble.')])
        words=[dict(text=t,start=i*.1,end=(i+1)*.1) for i,t in enumerate(['Stay','quiet','until','marble'])]
        response=dict(text='Stay quiet until marble.',segments=[dict(words=words)])
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);sf.write(folder/'server-input.wav',np.zeros(24000),24000,subtype='PCM_16')
            self.assertTrue(quality.reflected_words(folder,scene,response)['passed'])
            for bad in (dict(response,text='Stay quiet.'),dict(response,segments=[dict(words=words[:-1])]),
                        dict(response,segments=[dict(words=[dict(w,start=w['end']) for w in words])])):
                self.assertFalse(quality.reflected_words(folder,scene,bad)['passed'])

    def test_repair_cannot_be_applied_to_a_different_failure_or_passed_subset(self):
        for report in (dict(passed=True,results=[]),
                       dict(passed=False,results=[dict(id='other',checks=dict(passed=False))])):
            with patch.object(quality.parent,'verify',return_value=report):
                with self.assertRaisesRegex(ValueError,'Only diagnosed39/40'):
                    quality.parent_evidence(Path('raw'),Path('source'),Path('batch'),Path('quality'))


if __name__=='__main__':unittest.main()
