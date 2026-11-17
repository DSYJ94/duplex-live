import hashlib
import unittest
import numpy as np
from training.personaplex.interaction_preview import reconstruct


class InteractionPreviewTest(unittest.TestCase):
    def source(self):
        user=np.zeros(1920+60*24000,dtype=np.float32);user[:1920]=.1
        first=np.full(1920,.25,dtype=np.float32);second=np.full(1920,.5,dtype=np.float32)
        raw=np.concatenate([first,second]);trace=[];offset=0
        for index in range(len(user)//1920):
            pcm=first if index==2 else second if index==5 else raw[:0]
            trace.append(dict(sequence=index,raw_sample_offset=offset,audio_samples=len(pcm),
                              pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest()))
            offset+=len(pcm)
        return user,raw,trace

    def test_missing_output_frames_do_not_shift_native_reply_earlier(self):
        user,raw,trace=self.source();timeline=reconstruct(user,raw,trace,user_frames=1920)
        np.testing.assert_array_equal(timeline[:,1],user)
        self.assertFalse(timeline[:3840,0].any())
        np.testing.assert_array_equal(timeline[3840:5760,0],raw[:1920])
        self.assertFalse(timeline[5760:9600,0].any())
        np.testing.assert_array_equal(timeline[9600:11520,0],raw[1920:])

    def test_changed_audio_or_trace_cannot_be_presented_as_exact(self):
        user,raw,trace=self.source();raw[0]+=.1
        with self.assertRaises(ValueError):reconstruct(user,raw,trace,user_frames=1920)
        user,raw,trace=self.source();trace[5]['raw_sample_offset']=0
        with self.assertRaises(ValueError):reconstruct(user,raw,trace,user_frames=1920)


if __name__=='__main__':unittest.main()
