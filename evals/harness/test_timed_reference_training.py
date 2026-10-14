"""Test causal conditioning against the actual pinned native transformer."""
import io
import tempfile
import unittest
from types import SimpleNamespace

import torch
from moshi.models import LMModel
from moshi.utils.compile import no_compile
from evals.moshi_timed_reference import forward_with_reference, reference_schedule


def tiny_model():
    return LMModel(dim=16, n_q=2, dep_q=2, card=16, text_card=32,
        delays=[0, 0, 1], num_heads=2, num_layers=2, hidden_scale=2,
        depformer_dim=16, depformer_num_heads=2, depformer_num_layers=1,
        depformer_dim_feedforward=32, positional_embedding="rope", depformer_pos_emb="rope",
        context=64, device="cpu", dtype=torch.float32).eval()


class TimedReferenceTest(unittest.TestCase):
    def test_mixed_dtype_conditioner_adapter_reload_is_exact(self):
        from pathlib import Path
        from safetensors.torch import save_file
        from moshi.models.loaders import get_lora_moshi
        from evals.moshi_rag_adapter_pilot import load_adapter_exact, adapter_parameters
        from evals.moshi_training_smoke import tensor_hashes
        model = tiny_model().bfloat16()
        model.condition_provider = torch.nn.Module()
        speaker = torch.nn.Module()
        speaker.output_proj = torch.nn.Linear(16, 16, bias=False).float()
        model.condition_provider.conditioners = torch.nn.ModuleDict({"first_speaker": speaker})
        base = {n:t.clone() for n,t in model.state_dict().items()}
        model = get_lora_moshi(model, None, lora_rank=2, lora_scaling=2., device="cpu", fuse_lora=False)
        adapters = adapter_parameters(model)
        expected = tensor_hashes(adapters.items())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"lora.safetensors"
            save_file({n:p.detach().contiguous() for n,p in adapters.items()}, path)
            reloaded = tiny_model().bfloat16()
            reloaded.condition_provider = torch.nn.Module()
            speaker = torch.nn.Module()
            speaker.output_proj = torch.nn.Linear(16, 16, bias=False).float()
            reloaded.condition_provider.conditioners = torch.nn.ModuleDict({"first_speaker": speaker})
            reloaded.load_state_dict(base)
            reloaded = load_adapter_exact(reloaded, path, rank=2, scaling=2., device="cpu")
            self.assertEqual(expected, tensor_hashes(adapter_parameters(reloaded).items()))
            self.assertEqual(reloaded.condition_provider.conditioners["first_speaker"].output_proj.frozen_W.weight.dtype, torch.float32)
            for name, parameter in reloaded.named_parameters():
                original_name = name.replace(".frozen_W", "")
                if original_name in base:
                    torch.testing.assert_close(parameter, base[original_name], atol=0, rtol=0)

    def test_native_lora_zero_start_preserves_pretrained_outputs(self):
        from moshi.models.loaders import get_lora_moshi
        torch.manual_seed(49)
        model = tiny_model()
        codes = torch.randint(0, 16, (1, 3, 12))
        reference = torch.randn(1, 12, 16)
        with torch.no_grad(), no_compile():
            before = forward_with_reference(model, codes, reference)
            model = get_lora_moshi(model, None, lora_rank=2, lora_scaling=2., device="cpu", fuse_lora=False)
            for name, parameter in model.named_parameters():
                if ".lora_B." in name:
                    parameter.zero_()
            after = forward_with_reference(model, codes, reference)
            torch.testing.assert_close(before.text_logits, after.text_logits, equal_nan=True, atol=0, rtol=0)
            torch.testing.assert_close(before.logits, after.logits, equal_nan=True, atol=0, rtol=0)

    def test_speaker_prefix_matches_serving_and_keeps_target_alignment(self):
        from moshi.models.lm import _delay_sequence
        torch.manual_seed(47)
        model = tiny_model()
        codes = torch.randint(0, 16, (1, 3, 12))
        prefix = torch.randn(1, 2, 16)
        reference = torch.zeros(1, 12, 16)
        reference[:, 5:8] = torch.randn(1, 3, 16)
        model.fuser = SimpleNamespace(get_sum=lambda _: None, get_cross=lambda _: None,
            get_prepend=lambda _: prefix, get_streaming_sum=lambda _: None)
        initial = model._get_initial_token()
        delayed = _delay_sequence(model.delays, codes, initial)
        sequence = torch.cat([initial, delayed], dim=2)[..., :-1]
        with torch.no_grad(), no_compile():
            result = forward_with_reference(model, codes, reference, {"speaker": prefix})
            expected = []
            with model.streaming(1):
                model.forward_text(sequence_emb=prefix)
                for index in range(12):
                    _, logits = model.forward_text(sequence=sequence[..., index:index+1],
                        streaming_sum_condition=reference[:, index:index+1])
                    expected.append(logits)
            self.assertEqual(result.text_logits.shape, (1, 1, 12, 32))
            torch.testing.assert_close(result.text_logits, torch.cat(expected, dim=2), atol=2e-5, rtol=2e-5)
            self.assertEqual(result.logits.shape, (1, 2, 12, 16))
            without_prefix = forward_with_reference(model, codes, reference)
            self.assertGreater((result.text_logits-without_prefix.text_logits).abs().max().item(), 1e-5)
        model.train()
        model.transformer.checkpointing = True
        with no_compile():
            result = forward_with_reference(model, codes, reference, {"speaker": prefix})
            result.text_logits[result.text_mask].square().mean().backward()
            self.assertTrue(torch.isfinite(model.text_linear.weight.grad).all())

    def test_rows_wait_for_ready_and_stop_at_cancellation(self):
        times = torch.tensor([0., .08, .16, .24, .32, .40])
        result = reference_schedule(times, [{"ready_at_s": .17,
            "valid_until_s": .32, "rows": torch.ones(4, 16)}], 16)
        self.assertEqual(result[:3].abs().sum().item(), 0)
        self.assertEqual(result[3].sum().item(), 16)
        self.assertEqual(result[4:].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            reference_schedule(times, [{"ready_at_s": .1, "rows": torch.ones(4, 16)},
                {"ready_at_s": .2, "rows": torch.ones(2, 16)}], 16)

    def test_full_sequence_matches_real_one_frame_conditioning(self):
        torch.manual_seed(31)
        model = tiny_model()
        sequence = torch.randint(0, 16, (1, 3, 12))
        reference = torch.zeros(1, 12, 16)
        reference[:, 5:8] = torch.randn(1, 3, 16)
        with torch.no_grad(), no_compile():
            _, full = model.forward_text(sequence=sequence, sum_condition=reference)
            streamed = []
            with model.streaming(1):
                for index in range(12):
                    _, logits = model.forward_text(sequence=sequence[..., index:index+1],
                        streaming_sum_condition=reference[:, index:index+1])
                    streamed.append(logits)
            torch.testing.assert_close(full, torch.cat(streamed, dim=2), atol=2e-5, rtol=2e-5)
            changed = reference.clone()
            changed[:, 8:] = 100 * torch.randn(1, 4, 16)
            _, future_changed = model.forward_text(sequence=sequence, sum_condition=changed)
            torch.testing.assert_close(full[..., :8, :], future_changed[..., :8, :], atol=0, rtol=0)

    def test_real_training_backward_checkpointing_and_reload(self):
        torch.manual_seed(22)
        model = tiny_model()
        codes = torch.randint(0, 16, (1, 3, 12))
        reference = torch.zeros(1, 12, 16)
        reference[:, 5:8] = torch.randn(1, 3, 16)
        initial_keys = set(model.state_dict())
        with no_compile():
            baseline = model(codes=codes)
            zero = forward_with_reference(model, codes, torch.zeros_like(reference))
            torch.testing.assert_close(baseline.text_logits, zero.text_logits, equal_nan=True)
            result = forward_with_reference(model, codes, reference)
            torch.testing.assert_close(result.text_logits[..., :5, :], baseline.text_logits[..., :5, :],
                                       equal_nan=True, atol=0, rtol=0)
            self.assertGreater((result.text_logits[..., 5:, :] - baseline.text_logits[..., 5:, :]).abs().max().item(), 1e-5)
            loss = result.text_logits[result.text_mask].square().mean()
            loss.backward()
            expected_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            model.transformer.checkpointing = True
            model.depformer.checkpointing = True
            result = forward_with_reference(model, codes, reference)
            result.text_logits[result.text_mask].square().mean().backward()
            for name, parameter in model.named_parameters():
                if name in expected_grads:
                    torch.testing.assert_close(parameter.grad, expected_grads[name])
            self.assertTrue(all(torch.isfinite(g).all() for g in expected_grads.values()))
            self.assertTrue(any(g.abs().sum() > 0 for g in expected_grads.values()))
            self.assertEqual(initial_keys, set(model.state_dict()))
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            reloaded = tiny_model()
            reloaded.load_state_dict(torch.load(stream, weights_only=True))
            actual = forward_with_reference(reloaded, codes, reference)
            torch.testing.assert_close(result.text_logits, actual.text_logits, equal_nan=True)
            self.assertNotIn("forward_text", model.__dict__)


if __name__ == "__main__":
    unittest.main()
