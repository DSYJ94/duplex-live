import unittest
from evals.context_observed_teacher import observed_end


class ObservedTeacherTest(unittest.TestCase):
    def fixture(self):
        import torch
        x=torch.full((17,470),3,dtype=torch.int64);x[1:,-1]=-1
        return x

    def test_physical_integer_boundary_removes_only_proved_phantom_padding(self):
        self.assertEqual(observed_end(self.fixture(),900480,24000),469)

    def test_spoken_control_or_real_audio_cannot_be_dropped(self):
        for token in (4,77):
            x=self.fixture();x[0,-1]=token
            with self.assertRaises(RuntimeError):observed_end(x,900480,24000)
        x=self.fixture();x[1,-1]=19
        with self.assertRaises(RuntimeError):observed_end(x,900480,24000)
        x=self.fixture();x[8,20]=-1
        with self.assertRaises(RuntimeError):observed_end(x,900480,24000)

    def test_incomplete_source_frame_or_arbitrary_crop_is_rejected(self):
        with self.assertRaises(RuntimeError):observed_end(self.fixture(),900479,24000)
        with self.assertRaises(RuntimeError):observed_end(self.fixture(),898560,24000)


if __name__=='__main__':unittest.main()
