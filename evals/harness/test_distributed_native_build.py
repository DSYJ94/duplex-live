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
        self.transformer = nn.Module(); self.transformer.layers = nn.ModuleList([nn.Sequential(nn.Linear(4, 4))])
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
    def test_selective_ownership_and_audio_gradient_preserve_frozen_decoder(self):
        lm = ToyNative().bfloat16()
        lm.audio_emb = nn.Linear(4, 4).bfloat16()
        before = copy.deepcopy(lm.state_dict())
        report = configure_lm(lm, 'selective-trunk-text')
        expected = sorted(n for n in before if n.startswith(('transformer.', 'text_linear.')))
        self.assertEqual(report['trainable_names'], expected)
        self.assertEqual(report['trainable_parameters'], sum(dict(lm.named_parameters())[n].numel() for n in expected))
        self.assertEqual(report['total_parameters'], sum(p.numel() for p in lm.parameters()))
        for name, parameter in lm.named_parameters():
            torch.testing.assert_close(parameter, before[name].float(), rtol=0, atol=0)
            self.assertEqual(parameter.requires_grad, name in expected)
        dispatch = NativeDispatch(lm)
        audio = dispatch('audio', torch.randn(1, 3, 4))[0]
        audio.square().sum().backward()
        self.assertGreater(lm.transformer.layers[0][0].weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in lm.depformer.parameters()))
        optimizer = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=.01)
        optimizer.step()
        self.assertFalse(torch.equal(lm.transformer.layers[0][0].weight, before['transformer.layers.0.0.weight'].float()))
        for name in ('depformer.layers.0.weight', 'depformer.layers.0.bias', 'audio_emb.weight', 'audio_emb.bias'):
            torch.testing.assert_close(lm.state_dict()[name], before[name].float(), rtol=0, atol=0)
        optimizer.zero_grad(set_to_none=True)
        dispatch('text', torch.randn(1, 3, 4))[1].square().sum().backward()
        self.assertGreater(lm.text_linear.weight.grad.abs().sum(), 0)

    def test_selective_rejects_tied_acoustic_ownership_and_saved_adapter(self):
        lm = ToyNative()
        lm.depformer.layers[0].weight = lm.transformer.layers[0][0].weight
        with self.assertRaises(ValueError): configure_lm(lm, 'selective-trunk-text')
        lm = ToyNative()
        configure_lm(lm, 'broad-lora64')
        with self.assertRaises(ValueError): configure_lm(lm, 'selective-trunk-text')

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


    def test_real_gloo_adapter_save_restore_does_not_duplicate_frozen_base(self):
        import tempfile
        from pathlib import Path
        import torch.distributed as dist
        from training.personaplex.distributed_native import DistributedNative
        with tempfile.TemporaryDirectory() as folder:
            dist.init_process_group('gloo', init_method='file://' + str(Path(folder)/'rendezvous'), rank=0, world_size=1)
            try:
                initial=ToyNative(); other=copy.deepcopy(initial)
                checks=[]
                def configured(model,ownership):
                    self.assertIs(model,initial);self.assertGreater(ownership['trainable_parameters'],0);checks.append(True)
                runtime=DistributedNative(initial,'broad-lora64','cpu',after_configure=configured)
                self.assertEqual(checks,[True])
                opt=torch.optim.AdamW(runtime.parameters(),lr=1e-3)
                x=torch.randn(1,2,4)
                runtime.view.forward_codes(x)[1].square().mean().backward()
                runtime.clip_grad_norm(1.); opt.step()
                state=runtime.full_state(opt)
                self.assertTrue(all('.lora_' in n for n in state['model']))
                restored=DistributedNative(other,'broad-lora64','cpu')
                other_opt=torch.optim.AdamW(restored.parameters(),lr=1e-3)
                restored.load_full_state(state,other_opt)
                torch.testing.assert_close(runtime.view.forward_codes(x)[1],restored.view.forward_codes(x)[1],rtol=0,atol=0)
                self.assertEqual(len(other_opt.state_dict()['state']),len(opt.state_dict()['state']))
                invalid=dict(state,model={})
                with self.assertRaises(ValueError):restored.load_full_state(invalid,other_opt)
            finally: dist.destroy_process_group()


if __name__ == '__main__': unittest.main()
