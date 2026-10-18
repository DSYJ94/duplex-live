"""Check gradient ownership, causal-column exclusion and genuine silence targets."""
import unittest
from types import SimpleNamespace
import torch
from evals.context_native_joint_loss import loss


class NativeJointLossTests(unittest.TestCase):
    def sample(self):
        torch.manual_seed(42)
        text = torch.randn(1, 1, 4, 8, requires_grad=True)
        audio = torch.randn(1, 16, 4, 4, requires_grad=True)
        output = SimpleNamespace(text_logits=text, logits=audio,
            text_mask=torch.ones(1, 1, 4, dtype=torch.bool),
            mask=torch.ones(1, 16, 4, dtype=torch.bool))
        codes = torch.zeros(1, 17, 4, dtype=torch.int64)
        codes[0, 0] = torch.tensor([0, 5, 3, 0])
        mask = torch.tensor([[False, True, True, False]])
        return output, codes, mask

    def test_only_real_assistant_columns_receive_gradient(self):
        out, codes, mask = self.sample()
        value, audit = loss(out, codes, mask, mask)
        value.backward()
        self.assertGreater(float(out.logits.grad[:, :8, 1:3].abs().sum()), 0)
        self.assertEqual(float(out.logits.grad[:, 8:].abs().sum()), 0)
        for index in (0, 3):
            self.assertEqual(float(out.logits.grad[:, :, index].abs().sum()), 0)
            self.assertEqual(float(out.text_logits.grad[:, :, index].abs().sum()), 0)
        self.assertEqual(audit['assistant_audio_targets'], [2] * 8)
        self.assertEqual(audit['user_audio_loss_targets'], 0)

    def test_user_echo_and_masked_columns_cannot_change_target_loss(self):
        out, codes, mask = self.sample()
        first, _ = loss(out, codes, mask, mask)
        changed = codes.clone()
        changed[:, 9:] = 3
        changed[:, :, (0, 3)] = 9999
        with torch.no_grad():
            out.logits[:, 8:] = float('nan')
            out.logits[:, :, (0, 3)] = float('nan')
            out.text_logits[:, :, (0, 3)] = float('nan')
        second, _ = loss(out, changed, mask, mask)
        self.assertTrue(torch.equal(first, second))

    def test_silence_does_not_require_a_spoken_target(self):
        out, codes, mask = self.sample()
        codes[:, 0] = 3
        value, audit = loss(out, codes, mask, mask)
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(audit['spoken_targets'], 0)
        self.assertEqual(audit['clock_targets'], 2)

    def test_invalid_real_target_fails_closed(self):
        out, codes, mask = self.sample()
        codes[0, 1, 1] = 9999
        with self.assertRaises(ValueError):
            loss(out, codes, mask, mask)


if __name__ == '__main__':
    unittest.main()
