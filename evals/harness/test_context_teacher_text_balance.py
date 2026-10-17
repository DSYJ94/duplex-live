import unittest
from evals.context_teacher_text_balance import validate


class BalancePolicyTests(unittest.TestCase):
    def test_requires_complete_native_approval(self):
        validate('legacy', False, 'four-longest')
        validate('balanced-spoken', True, 'all-approved')
        for mode, native, coverage in [('unknown', True, 'all-approved'),
                ('balanced-spoken', False, 'all-approved'),
                ('balanced-spoken', True, 'four-longest')]:
            with self.assertRaises(ValueError): validate(mode, native, coverage)


class BalanceGradientTests(unittest.TestCase):
    def test_spoken_gradient_mass_is_not_diluted_by_silence_length(self):
        import torch as t
        from evals.context_teacher_text_balance import loss
        for silence in (2, 200):
            z = t.zeros(silence + 1, 8, requires_grad=True)
            y = t.tensor([5] + [3] * silence); m = t.ones_like(y, dtype=t.bool)
            value, report = loss(z, y, m); value.backward()
            # Equal predictions imply equal absolute gradient mass per group.
            self.assertAlmostEqual(float(z.grad[0].abs().sum()),
                float(z.grad[1:].abs().sum()), places=5)
            self.assertEqual(report['spoken_targets'], 1)
            self.assertEqual(report['clock_control_targets'], silence)
            self.assertAlmostEqual(float(value), float(t.tensor(8.).log()), places=6)

    def test_excluded_prefix_span_and_invalid_labels_stay_excluded(self):
        import torch as t
        from evals.context_teacher_text_balance import loss
        z = t.zeros(5, 8, requires_grad=True)
        y = t.tensor([-1, 5, -1, 3, 4]); m = t.tensor([False, True, False, True, True])
        value, report = loss(z, y, m)
        poisoned = z.detach().clone(); poisoned[~m] = float('nan')
        self.assertTrue(t.equal(value.detach(), loss(poisoned, y, m)[0]))
        value.backward(); self.assertEqual(int(t.count_nonzero(z.grad[~m])), 0)
        self.assertGreater(float(z.grad[4].abs().sum()), 0.)  # RET remains supervised.
        self.assertAlmostEqual(float(z.grad[4].abs().sum()),
            2. * float(z.grad[3].abs().sum()), places=6)
        self.assertEqual(report['user_loss_targets'], 0)
        for bad in (t.tensor([-1, 8, -1, 3, 4]), t.tensor([-1, 5, -1, -1, 4])):
            with self.assertRaises(RuntimeError): loss(z, bad, m)

    def test_missing_group_and_nonfinite_eligible_logits_fail_closed(self):
        import torch as t
        from evals.context_teacher_text_balance import loss
        for labels in ([5, 6], [3, 0]):
            with self.assertRaises(RuntimeError):
                loss(t.zeros(2, 8), t.tensor(labels), t.tensor([True, True]))
        z = t.zeros(2, 8); z[0, 0] = float('nan')
        with self.assertRaises(RuntimeError): loss(z, t.tensor([5, 3]), t.tensor([True, True]))


if __name__ == '__main__': unittest.main()
