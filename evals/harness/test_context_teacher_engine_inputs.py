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
        prefix=current[0,:,:1].clone()
        self.assertFalse(provided_ownership(flags,current,clean,True,prefix))
        flags[:,2:9,0]=True
        self.assertTrue(provided_ownership(flags,current,clean,True,prefix))
        wrong=prefix.clone();wrong[2,0]+=1
        self.assertFalse(provided_ownership(flags,current,clean,True,wrong))
        flags[:,2,2]=True
        self.assertFalse(provided_ownership(flags,current,clean,True,prefix))
        flags[:,2,2]=False
        flags[:,9,1]=False
        self.assertFalse(provided_ownership(flags,current,clean,True,prefix))
        flags[:,9,1]=True;current[0,0,2]+=1
        self.assertFalse(provided_ownership(flags,current,clean,True,prefix))

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

    def test_streamed_capture_retains_failed_offline_control_and_exact_native_bar(self):
        import torch
        from evals.context_teacher_engine_inputs import streamed_capture_control
        legacy=torch.zeros(1,17,9,dtype=torch.int64)
        actual=legacy.clone();actual[0,16,4]=7
        result=streamed_capture_control(actual,legacy,actual.clone(),True)
        self.assertTrue(result['passed'])
        self.assertFalse(result['legacy_offline_comparison']['exact'])
        self.assertTrue(result['streamed_user_input_comparison']['exact'])
        self.assertFalse(result['training_approved'])
        for change in ('assistant','native','ownership','frames'):
            a=actual.clone();streamed=actual.clone();owned=True
            if change=='assistant':a[0,8,3]=1;streamed=a.clone()
            elif change=='native':streamed[0,15,2]=1
            elif change=='ownership':owned=False
            else:a=a[:,:,:-1]
            with self.subTest(change=change):
                self.assertFalse(streamed_capture_control(a,legacy,streamed,owned)['passed'])


if __name__=='__main__':unittest.main()
