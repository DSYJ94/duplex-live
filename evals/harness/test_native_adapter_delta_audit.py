import unittest
import torch
from evals.native_adapter_delta_audit import effective_delta,product_norm


class AdapterDeltaMath(unittest.TestCase):
    def test_matrix_free_result_matches_independently_materialized_dense_matrices(self):
        rng=torch.Generator().manual_seed(71)
        for rank in (1,3,8):
            a=torch.randn(rank,19,generator=rng,dtype=torch.float64)
            b=torch.randn(23,rank,generator=rng,dtype=torch.float64)
            aa=a+1e-6*torch.randn(a.shape,generator=rng,dtype=torch.float64)
            bb=b+1e-6*torch.randn(b.shape,generator=rng,dtype=torch.float64)
            r=effective_delta(a,b,aa,bb,2.)
            dense_old=2*b@a;dense_new=2*bb@aa;delta=dense_new-dense_old
            self.assertAlmostEqual(r['parent_effective_LoRA_frobenius'],float(dense_old.norm()),places=10)
            self.assertAlmostEqual(r['delta_effective_LoRA_frobenius'],float(delta.norm()),places=12)
            self.assertAlmostEqual(r['parent_component_delta_cosine'],float((dense_old*delta).sum()/(dense_old.norm()*delta.norm())),places=9)

    def test_identical_weights_have_exact_zero_delta_even_when_parent_is_nonzero(self):
        a=torch.arange(12,dtype=torch.float32).reshape(3,4);b=torch.arange(15,dtype=torch.float32).reshape(5,3)
        r=effective_delta(a,b,a,b,2.)
        self.assertEqual(r['delta_effective_LoRA_frobenius'],0.)
        self.assertEqual(r['relative_to_parent_LoRA_component'],0.)
        self.assertIsNone(r['parent_component_delta_cosine'])

    def test_topology_or_nonfinite_input_rejects(self):
        a=torch.ones(2,4);b=torch.ones(5,2)
        with self.assertRaises(ValueError):effective_delta(a,b,a[:1],b,2.)
        a[0,0]=float('nan')
        with self.assertRaises(ValueError):effective_delta(a,b,a,b,2.)


if __name__=='__main__':unittest.main()
