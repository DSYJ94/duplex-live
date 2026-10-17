import unittest


class TeacherReferenceTest(unittest.TestCase):
    def test_reference_cannot_wrap_gradient_training(self):
        import torch
        from evals.context_teacher_numeric_reference import linear_reference
        with torch.enable_grad(),self.assertRaises(RuntimeError):
            with linear_reference():pass

    def test_global_linear_is_restored_even_when_audit_fails(self):
        import torch
        from torch.nn import functional as F
        from evals.context_teacher_numeric_reference import linear_reference
        original=F.linear;weight=torch.randn(3,5);saved=weight.clone()
        with torch.no_grad(),self.assertRaisesRegex(RuntimeError,'audit failure'):
            with linear_reference():
                self.assertIsNot(F.linear,original)
                result=F.linear(torch.randn(2,5),weight)
                self.assertEqual(result.dtype,weight.dtype)
                self.assertTrue(torch.equal(weight,saved))
                raise RuntimeError('audit failure')
        self.assertIs(F.linear,original)


if __name__=='__main__':unittest.main()
