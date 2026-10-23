"""Exercise actual autograd ownership and acoustic/text gradient isolation."""
import unittest
import torch
from evals.native_control_gradient import gradient_summary, output_hashes
from types import SimpleNamespace


class ControlGradient(unittest.TestCase):
    def model(self):
        model = torch.nn.Module()
        model.temporal = torch.nn.Parameter(torch.tensor([1., 2.]))
        model.text_linear = torch.nn.Module()
        model.text_linear.lora_A = torch.nn.Parameter(torch.tensor([2.]))
        model.text_linear.lora_B = torch.nn.Parameter(torch.tensor([3.]))
        model.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        learned = {n: p for n, p in model.named_parameters() if p.requires_grad}
        (model.temporal.sum() * model.text_linear.lora_A.sum() + model.text_linear.lora_B.sum()).backward()
        return model, learned

    def test_acoustic_update_must_change_temporal_gradient_and_preserve_text_head_exactly(self):
        model, learned = self.model()
        previous = {n: p.grad.clone() for n, p in learned.items()}
        with self.assertRaisesRegex(ValueError, 'no finite temporal gradient'):
            gradient_summary(model, learned, previous)
        model.temporal.square().sum().backward()
        result = gradient_summary(model, learned, previous)
        self.assertTrue(result['acoustic_text_head_gradient_delta_exactly_zero'])
        self.assertGreater(result['acoustic_temporal_gradient_delta_norm'], 0.)
        model.text_linear.lora_A.sum().backward()
        with self.assertRaisesRegex(ValueError, 'text-head gradient'):
            gradient_summary(model, learned, previous)

    def test_missing_nonfinite_and_frozen_gradients_are_rejected(self):
        for mode in ('missing', 'nonfinite', 'frozen'):
            model, learned = self.model()
            if mode == 'missing': model.temporal.grad = None
            if mode == 'nonfinite': model.temporal.grad[0] = float('nan')
            if mode == 'frozen': model.frozen.grad = torch.ones_like(model.frozen)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                gradient_summary(model, learned)

    def test_hashes_cover_both_audio_text_and_their_masks(self):
        values = dict(logits=torch.ones(1, 8, 2, 3), mask=torch.ones(1, 8, 2, dtype=torch.bool),
                      text_logits=torch.ones(1, 1, 2, 3), text_mask=torch.ones(1, 1, 2, dtype=torch.bool))
        reference = output_hashes(SimpleNamespace(**values))
        for name in values:
            changed = {k: v.clone() for k, v in values.items()}
            changed[name].flatten()[0] = 0
            with self.subTest(name=name):
                self.assertNotEqual(output_hashes(SimpleNamespace(**changed)), reference)


if __name__ == '__main__': unittest.main()
