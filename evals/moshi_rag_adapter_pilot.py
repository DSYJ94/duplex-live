"""Native LoRA optimizer and fresh-reload proof on a short approved sample."""
import json
import sys
from pathlib import Path


def adapter_parameters(model):
    return {n: p for n, p in model.named_parameters() if ".lora_A." in n or ".lora_B." in n}


def precise_sample(model, codes, reference):
    import torch
    from moshi.utils.compile import no_compile
    from moshi.inference_utils.utils import get_condition_tensors
    from evals.moshi_timed_reference import forward_with_reference
    parameters = [(p, p.dtype) for p in model.parameters()]
    flags = (model.training, model.transformer.checkpointing, model.depformer.checkpointing,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        model.eval()
        for p, _ in parameters:
            p.data = p.data.float()
        model.transformer.checkpointing = model.depformer.checkpointing = False
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode(), no_compile():
            conditions = get_condition_tensors("moshi", model, 1, 1., first_speaker="user")
            result = forward_with_reference(model, codes, reference, conditions)
            return {"text": result.text_logits.float().cpu(), "audio": result.logits.float().cpu()}
    finally:
        for p, dtype in parameters:
            p.data = p.data.to(dtype)
        model.train(flags[0])
        model.transformer.checkpointing, model.depformer.checkpointing = flags[1:3]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[3:]


def run_pilot(model, codes, reference, out, steps):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from moshi.models.loaders import get_lora_moshi
    from moshi.utils.compile import no_compile
    from moshi.inference_utils.utils import get_condition_tensors
    from evals.moshi_training_smoke import tensor_hashes
    from evals.moshi_timed_reference import forward_with_reference
    from evals.moshi_model_download import sha256
    trainer = Path("/projects/u6si/duplex-live/vendor/moshi-finetune-2acc879fe7c48f885a18f6cc9548bccb2674d87b")
    sys.path.insert(0, str(trainer))
    from finetune.mixed_precision import prepare_mixed_precision, upcast_mixed_precision, downcast_mixed_precision
    if not 1 <= steps <= 16:
        raise ValueError("Engineering pilot requires 1–16 optimizer steps")
    out.mkdir(parents=True, exist_ok=False)
    baseline = precise_sample(model, codes, reference)
    torch.manual_seed(424242)
    model = get_lora_moshi(model, None, lora_rank=8, lora_scaling=2., device="cuda", fuse_lora=False)
    model.requires_grad_(False)
    all_adapters = adapter_parameters(model)
    with torch.no_grad():
        for n, p in all_adapters.items():
            if ".lora_B." in n:
                p.zero_()
            p.requires_grad_(not n.startswith("condition_provider."))
    zero = precise_sample(model, codes, reference)
    if any(not torch.equal(torch.isnan(baseline[k]), torch.isnan(zero[k])) for k in zero):
        raise RuntimeError("Zero adapters changed pretrained logit masks")
    zero_error = {k: float(torch.nan_to_num((baseline[k]-zero[k]).abs()).max()) for k in zero}
    if any(v > .001 for v in zero_error.values()):
        raise RuntimeError(f"Zero adapters changed pretrained logits: {zero_error}")
    trainable = {n: p for n, p in all_adapters.items() if p.requires_grad}
    initial = tensor_hashes(all_adapters.items())
    frozen = tensor_hashes((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    prepare_mixed_precision(trainable.values(), param_dtype=torch.bfloat16, optim_dtype=torch.float32)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=1e-5, weight_decay=.01)
    model.train()
    model.transformer.checkpointing = model.depformer.checkpointing = True
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        conditions = get_condition_tensors("moshi", model, 1, 1., first_speaker="user")
        with no_compile():
            result = forward_with_reference(model, codes, reference, conditions)
            text = F.cross_entropy(result.text_logits[result.text_mask].float(), codes[:, :1][result.text_mask])
            targets = codes[:, model.audio_offset:model.audio_offset + model.dep_q]
            speech = F.cross_entropy(result.logits[result.mask].float(), targets[result.mask])
            loss = text + speech
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite conditioned training loss")
            loss.backward()
        gradients = [p.grad for p in trainable.values() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients) or not any(g.count_nonzero() for g in gradients):
            raise RuntimeError("Missing/nonfinite/zero adapter gradients")
        upcast_mixed_precision(model.parameters(), optim_dtype=torch.float32)
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient norm")
        optimizer.step()
        downcast_mixed_precision(model.parameters(), param_dtype=torch.bfloat16)
        losses.append({"step": step+1, "text_loss": float(text.detach()), "speech_loss": float(speech.detach()),
                       "gradient_norm": float(norm)})
        print(json.dumps(losses[-1]), flush=True)
    if frozen != tensor_hashes((n, p) for n, p in model.named_parameters() if not p.requires_grad):
        raise RuntimeError("Frozen pretrained/conditioner weights changed")
    final = tensor_hashes(all_adapters.items())
    changed = sum(initial[n] != digest for n, digest in final.items())
    if not changed:
        raise RuntimeError("No adapters changed")
    save_file({n: p.detach().cpu().contiguous() for n, p in all_adapters.items()}, out / "lora.safetensors")
    torch.save({"optimizer": optimizer.state_dict(), "master_parameters": {n: p._mp_param.detach().cpu() for n,p in trainable.items()},
                "steps": steps, "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, out / "optimizer.pt")
    sample = precise_sample(model, codes, reference)
    torch.save({"codes": codes.cpu(), "reference": reference.cpu(), **sample}, out / "fixed-reference.pt")
    report = {"passed": True, "mode": "conditioned adapter optimizer engineering pilot",
        "rank": 8, "scaling": 2., "optimizer_steps": steps, "losses": losses,
        "trainable_count": sum(p.numel() for p in trainable.values()), "adapter_keys": list(all_adapters),
        "trainable_keys": list(trainable), "adapter_hashes": final, "changed_adapter_tensors": changed,
        "zero_adapter_logit_errors": zero_error, "frozen_weights_unchanged": True,
        "saved_adapter_sha256": sha256(out / "lora.safetensors"), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "reference_precision": "float32, TF32 off, checkpointing off, no_compile",
        "quality_improvement_tested": False, "fresh_holdout": False}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_reload(model, out):
    import torch
    from moshi.models.loaders import get_lora_moshi
    from safetensors.torch import load_file
    from evals.moshi_training_smoke import tensor_hashes
    from evals.moshi_model_download import sha256
    report = json.loads((out / "report.json").read_text())
    if sha256(out / "lora.safetensors") != report["saved_adapter_sha256"]:
        raise RuntimeError("Saved adapter file changed")
    saved = load_file(out / "lora.safetensors")
    if tensor_hashes(saved.items()) != report["adapter_hashes"]:
        raise RuntimeError("Saved adapters differ from trained values")
    model = get_lora_moshi(model, out / "lora.safetensors", lora_rank=8, lora_scaling=2.,
                           device="cuda", dtype=torch.bfloat16, fuse_lora=False)
    if set(adapter_parameters(model)) != set(report["adapter_keys"]) or any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Reloaded adapter coverage is incomplete")
    if tensor_hashes(adapter_parameters(model).items()) != report["adapter_hashes"]:
        raise RuntimeError("Reloaded adapter values differ")
    state = torch.load(out / "fixed-reference.pt", map_location="cpu", weights_only=True)
    actual = precise_sample(model, state["codes"].cuda(), state["reference"].cuda())
    errors = {}
    for name, value in actual.items():
        expected = state[name]
        if not torch.equal(torch.isnan(value), torch.isnan(expected)):
            raise RuntimeError("Reload changed logit masks")
        valid = torch.isfinite(expected)
        if not torch.isfinite(value[valid]).all():
            raise RuntimeError("Reloaded logits nonfinite")
        errors[name] = float((value[valid]-expected[valid]).abs().max())
    if any(v > .001 for v in errors.values()):
        raise RuntimeError(f"Fresh reload differs: {errors}")
    result = {"passed": True, "fresh_process": True, "max_abs_logit_errors": errors,
              "saved_adapter_sha256": report["saved_adapter_sha256"]}
    (out / "reload-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
