import unittest
from evals.context_native_teacher_objective import validate_base_retention


class BaseRetentionTest(unittest.TestCase):
    def test_default_keeps_original_objective(self):
        self.assertEqual(validate_base_retention(1.,False),1.)
        self.assertEqual(validate_base_retention(1.,True),1.)

    def test_removal_requires_native_teacher_gate(self):
        self.assertEqual(validate_base_retention(0.,True),0.)
        with self.assertRaises(ValueError):validate_base_retention(0.,False)

    def test_undeclared_intermediate_nonfinite_or_boolean_settings_fail(self):
        for value in (-1.,.5,2.,float('nan'),float('inf'),False):
            with self.assertRaises(ValueError):validate_base_retention(value,True)


if __name__=='__main__':unittest.main()
