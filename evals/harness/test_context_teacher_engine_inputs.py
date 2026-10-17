import unittest
from evals.context_teacher_engine_inputs import compare


class TeacherEngineInputTest(unittest.TestCase):
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
