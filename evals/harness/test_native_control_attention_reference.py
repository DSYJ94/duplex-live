import unittest
import torch
from torch.nn import functional as F
from evals.native_control_attention_reference import attention_reference


class AttentionReference(unittest.TestCase):
    def test_reference_matches_independent_double_math_and_preserves_native_call_shapes(self):
        generator=torch.Generator().manual_seed(27)
        q=torch.randn(1,2,12,8,generator=generator);k=torch.randn(1,2,12,8,generator=generator);v=torch.randn(1,2,12,8,generator=generator)
        original=F.scaled_dot_product_attention
        with torch.no_grad(),attention_reference() as stats:
            full=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            each=torch.cat([F.scaled_dot_product_attention(q[:,:,i:i+1],k[:,:,:i+1],v[:,:,:i+1]) for i in range(12)],dim=2)
            expected=original(q.double(),k.double(),v.double(),is_causal=True).float()
            self.assertTrue(torch.equal(full,expected));self.assertTrue(torch.equal(full,each))
            self.assertEqual(full.dtype,torch.float32);self.assertEqual(stats['calls'],13)
        self.assertIs(F.scaled_dot_product_attention,original)

    def test_mask_is_preserved_and_production_dtype_grad_or_dropout_cannot_enter(self):
        q=torch.ones(1,1,3,4);mask=torch.zeros(3,3);mask[:,2]=-float('inf')
        original=F.scaled_dot_product_attention
        with self.assertRaises(ValueError),attention_reference():pass
        with torch.no_grad(),attention_reference():
            actual=F.scaled_dot_product_attention(q,q,q,mask)
            self.assertTrue(torch.equal(actual,original(q.double(),q.double(),q.double(),mask.double()).float()))
            with self.assertRaises(ValueError):F.scaled_dot_product_attention(q,q,q,dropout_p=.1)
            with self.assertRaises(ValueError):F.scaled_dot_product_attention(q.half(),q.half(),q.half())
        self.assertIs(F.scaled_dot_product_attention,original)


if __name__=='__main__':unittest.main()
