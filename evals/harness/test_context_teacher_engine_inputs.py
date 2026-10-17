import unittest
from evals.context_teacher_engine_inputs import compare


class TeacherEngineInputTest(unittest.TestCase):
    def test_native_audio_never_supplies_agent_acoustic_tokens(self):
        import torch as t
        from evals.context_teacher_engine_inputs import forcing,provided_ownership
        clean=t.arange(17*4).reshape(17,4)
        self.assertEqual(set(forcing(clean,2,'cpu',True)),{'text_token'})
        legacy=forcing(clean,2,'cpu',False)
        self.assertTrue(t.equal(legacy['moshi_tokens'],clean[None,1:9,2:3]))
        flags=t.ones(1,17,4,dtype=t.bool);current=clean[None].clone()
        self.assertTrue(provided_ownership(flags,current,clean,False))
        self.assertFalse(provided_ownership(flags,current,clean,True))
        flags[:,1:9]=False
        self.assertTrue(provided_ownership(flags,current,clean,True))
        flags[:,9,1]=False
        self.assertFalse(provided_ownership(flags,current,clean,True))
        flags[:,9,1]=True;current[0,0,2]+=1
        self.assertFalse(provided_ownership(flags,current,clean,True))

    def test_one_changed_speaker_token_or_missing_frame_fails_parity(self):
        import torch
        expected=torch.arange(17*8,dtype=torch.int64).reshape(1,17,8)
        self.assertTrue(compare(expected.clone(),expected)['exact'])
        for channel in (0,1,8,9,16):
            changed=expected.clone();changed[0,channel,3]+=1
            result=compare(changed,expected)
            self.assertFalse(result['exact']);self.assertEqual(sum(result['mismatches_per_channel']),1)
            self.assertEqual(result['first_mismatches'][0]['channel'],channel)
        self.assertFalse(compare(expected[:,:,:-1],expected)['exact'])

    def test_user_codec_comparison_uses_all_eight_codebooks(self):
        import torch
        expected=torch.zeros(1,8,5,dtype=torch.int64);actual=expected.clone();actual[0,7,4]=12
        self.assertFalse(compare(actual,expected)['exact'])


if __name__=='__main__':unittest.main()
