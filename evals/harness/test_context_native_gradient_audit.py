"""Diagnostic arithmetic and complete schedule coverage, never behavior qualification."""
import unittest
from evals.context_native_gradient_audit import schedules, cosine


class NativeGradientAuditTests(unittest.TestCase):
    def test_exact_teacher_schedule_includes_quiet_and_pause_by32(self):
        kinds=['k'+str(i%8) for i in range(28)]
        old=['private']*8+['simple']*8+['clarify']*5+['cancel']*4
        new=['private']*8+['simple']*8+['clarify']*4+['cancel']*4
        plan=schedules(kinds,old,new)
        self.assertEqual([i for i,n in enumerate(plan['8']['new24']) if n],[0,1,8,9,16,17,20,21])
        self.assertTrue(all(plan['32']['new24']))
        self.assertEqual(plan['8']['new24'][14:16],[0,0])
        for length,counts in plan.items():
            for values in counts.values():self.assertEqual(sum(values),int(length))

    def test_opposed_gradient_and_zero_reference_are_distinct(self):
        import torch
        x=torch.tensor([3.,4.]);y=torch.tensor([-6.,-8.])
        self.assertAlmostEqual(cosine(x,y),-1.)
        self.assertAlmostEqual(cosine(x,torch.tensor([4.,-3.])),0.)
        self.assertIsNone(cosine(x,torch.zeros(2)))


if __name__=='__main__':unittest.main()
