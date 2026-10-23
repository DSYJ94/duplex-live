import unittest
import torch
from torch.nn import functional as F
from evals.native_control_reference import linear_reference as original
from evals.native_control_cached_reference import linear_reference


class CachedReference(unittest.TestCase):
    def test_exact_original_outputs_for_repeated_native_call_shapes(self):
        torch.manual_seed(37);w=torch.randn(19,32);b=torch.randn(19)
        inputs=[torch.randn(n,32) for n in (1,7,1,37,1)]
        with torch.no_grad(),original():expected=[F.linear(x,w,b) for x in inputs]
        with torch.no_grad(),linear_reference(100000) as stats:
            actual=[F.linear(x,w,b) for x in inputs]
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,actual)))
        self.assertEqual(stats['cast_cache_hits'],8)
        self.assertEqual(stats['cached_fp64_weight_bytes'],(w.numel()+b.numel())*8)

    def test_limit_falls_back_to_exact_temporary_casts(self):
        w=torch.randn(19,32);x=torch.randn(7,32)
        with torch.no_grad(),original():expected=F.linear(x,w)
        with torch.no_grad(),linear_reference(1) as stats:actual=F.linear(x,w)
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(stats['cached_fp64_weight_bytes'],0)

    def test_changed_weight_cannot_reuse_a_stale_cast(self):
        w=torch.ones(3,2);x=torch.ones(1,2);before=F.linear
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'changed'):
            with linear_reference(10000):
                F.linear(x,w);w.add_(1);F.linear(x,w)
        self.assertIs(F.linear,before)

    def test_rejects_gradients_bf16_and_nested_reference(self):
        with self.assertRaises(ValueError):
            with linear_reference():pass
        with torch.no_grad(),original(),self.assertRaisesRegex(ValueError,'nest'):
            with linear_reference():pass
        with torch.no_grad(),self.assertRaisesRegex(ValueError,'FP32'):
            with linear_reference():F.linear(torch.ones(1,2,dtype=torch.bfloat16),torch.ones(3,2,dtype=torch.bfloat16))


if __name__=='__main__':unittest.main()
