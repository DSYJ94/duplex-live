"""CPU integration checks; no native model weights or GPU admission claimed."""
import copy
import unittest
from contextlib import nullcontext
import torch
from torch import nn
from training.personaplex.distributed_native import NativeDispatch, NativeView, configure_lm


class ToyNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module(); self.transformer.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.depformer = nn.Module(); self.depformer.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.text_linear = nn.Linear(4, 8)
    def forward_codes(self, x):
        h = self.transformer.layers[0](x)
        return h, self.text_linear(h)
    def forward_train(self, x):
        h, z = self.forward_codes(x)
        return self.depformer.layers[0](h), z
    def streaming(self, batch):
        return nullcontext()


class NativeBuild(unittest.TestCase):
    def test_full_ownership_preserves_initial_values_and_has_audio_gradients(self):
        lm = ToyNative().bfloat16(); before = copy.deepcopy(lm.state_dict())
        report = configure_lm(lm, 'full-lm')
        self.assertFalse(report['codec_included']); self.assertEqual(report['details']['new_model_weights'], 0)
        for n, p in lm.named_parameters():
            self.assertTrue(p.requires_grad); self.assertEqual(p.dtype, torch.float32)
            torch.testing.assert_close(p, before[n].float(), rtol=0, atol=0)
        NativeDispatch(lm)('audio', torch.randn(1, 3, 4))[0].square().sum().backward()
        self.assertGreater(lm.depformer.layers[0].weight.grad.abs().sum(), 0)

    def test_native_calls_enter_outer_forward_hook(self):
        lm = ToyNative(); wrapped = NativeDispatch(lm); calls = []
        handle = wrapped.register_forward_pre_hook(lambda module, args: calls.append(args[0]))
        view = NativeView(wrapped, lm); x = torch.randn(1, 2, 4)
        torch.testing.assert_close(view.forward_codes(x)[1], lm.forward_codes(x)[1])
        torch.testing.assert_close(view.forward_train(x)[0], lm.forward_train(x)[0])
        self.assertEqual(calls, ['text', 'audio']); self.assertIs(view.depformer, lm.depformer)
        with view.streaming(1): pass
        handle.remove()

    def test_adapter_zero_function_and_exact_frozen_ownership(self):
        lm = ToyNative(); x = torch.randn(1, 2, 4); before = lm.forward_codes(x)[1].detach()
        report = configure_lm(lm, 'broad-lora64')
        torch.testing.assert_close(lm.forward_codes(x)[1], before, rtol=0, atol=0)
        self.assertTrue(all('.lora_' in n for n, p in lm.named_parameters() if p.requires_grad))
        self.assertGreater(report['trainable_parameters'], 0)
        with self.assertRaises(ValueError): configure_lm(lm, 'full-lm')

    def test_codec_and_unknown_dispatch_rejected(self):
        lm = ToyNative(); lm.mimi = nn.Linear(1, 1)
        with self.assertRaises(ValueError): configure_lm(lm, 'full-lm')
        with self.assertRaises(ValueError): NativeDispatch(ToyNative())('bad', torch.ones(1, 4))


if __name__ == '__main__': unittest.main()
