"""Preserve existing reference arithmetic while avoiding a full FP64 model copy."""
import unittest
import torch
from torch.nn import functional as F
from evals.context_teacher_numeric_reference import linear_reference as original_reference
from evals.native_control_reference import linear_reference


class ControlReference(unittest.TestCase):
    def test_bitwise_cpu_outputs_equal_original_reference_for_each_call_shape(self):
        torch.manual_seed(74)
        weights=[torch.randn(13,32),torch.randn(7,13),torch.randn(5,7)]
        biases=[torch.randn(w.shape[0]) for w in weights]
        before=[w.clone() for w in weights]
        for shape in ((1,32),(7,32),(2,9,32)):
            x=torch.randn(shape)
            with torch.no_grad(),original_reference():
                expected=x
                for w,b in zip(weights,biases):expected=F.linear(expected,w,b)
            with torch.no_grad(),linear_reference() as stats:
                actual=x
                for w,b in zip(weights,biases):actual=F.linear(actual,w,b)
            self.assertTrue(torch.equal(expected,actual))
            self.assertEqual(stats['calls'],3)
            self.assertEqual(stats['cached_fp64_weight_bytes'],0)
            self.assertEqual(stats['maximum_single_fp64_weight_bytes'],13*32*8)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(weights,before)))

    def test_rejects_training_and_production_bf16_and_restores_after_error(self):
        original=F.linear
        with self.assertRaises(ValueError):
            with linear_reference():pass
        with torch.no_grad():
            with self.assertRaisesRegex(ValueError,'FP32'):
                with linear_reference():F.linear(torch.ones(1,2,dtype=torch.bfloat16),torch.ones(3,2,dtype=torch.bfloat16))
        self.assertIs(F.linear,original)

    def test_does_not_silently_nest_an_existing_reference(self):
        with torch.no_grad(),original_reference():
            with self.assertRaisesRegex(ValueError,'nest'):
                with linear_reference():pass


if __name__=='__main__':unittest.main()
