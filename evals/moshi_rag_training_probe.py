"""Full released-weight backward proof, not a quality-training experiment."""
import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from evals.moshi_model_download import sha256


def main():
    import torch
    import torch.nn.functional as F
    from moshi.models import LMModel, loaders
    from moshi.inference_utils.utils import get_condition_tensors, get_conditioning_remote_async
    from moshi.utils.compile import no_compile
    from evals.moshi_training_smoke import validate_data
    from evals.moshi_timed_reference import forward_with_reference, reference_schedule

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    validate_data(args.data)
    manifest = json.loads(args.manifest.read_text())
    rag = Path(manifest["rag"]["path"])
    native = Path(manifest["native"]["path"])
    # The saved approved codes may be reused only with the identical codec and vocabulary.
    for name in ("tokenizer-e351c8d8-checkpoint125.safetensors", "tokenizer_spm_32k_3.model"):
        if sha256(rag / name) != sha256(native / name):
            raise RuntimeError("Native pilot codes are incompatible with this RAG checkpoint")
    state_path = args.native_run / "fixed-logits.pt"
    if sha256(state_path) != "e9d9694a4eb012450dc09682281e85ad807fb09fe7b92ab3c14b4c8e8e2efbb7":
        raise RuntimeError("Verified engineering code sample changed")
    codes = torch.load(state_path, map_location="cpu", weights_only=True)["codes"]
    info = loaders.CheckpointInfo.from_hf_repo(hf_repo=None,
        moshi_weights=rag / "model.safetensors", mimi_weights=rag / "tokenizer-e351c8d8-checkpoint125.safetensors",
        tokenizer=rag / "tokenizer_spm_32k_3.model", config_path=rag / "config.json")
    original = LMModel.load_state_dict
    audit = []

    def strict_speech_load(module, *pos, **kw):
        loaded = original(module, *pos, **kw)
        extras = [key for key in loaded.unexpected_keys if not key.startswith(
            "condition_provider.conditioners.reference_with_time.")]
        if loaded.missing_keys or extras:
            raise RuntimeError("Full speech checkpoint load mismatch")
        audit.append({"missing": [], "separately_loaded_reference_keys": len(loaded.unexpected_keys)})
        return loaded

    LMModel.load_state_dict = strict_speech_load
    try:
        model = info.get_moshi(device="cuda", dtype=torch.bfloat16,
                               skip_conditioners=["reference_with_time"])
    finally:
        LMModel.load_state_dict = original
    if len(audit) != 1:
        raise RuntimeError("Missing weight-load audit")
    model.requires_grad_(False)
    if args.reload:
        from evals.moshi_rag_adapter_pilot import verify_reload
        verify_reload(model, args.out)
        return
    # Trainable scope here is only a backward engineering check. No optimizer
    # or weights are saved, and this is not the selected adaptation recipe.
    layers = sorted({int(n.split(".")[2]) for n, _ in model.named_parameters()
                     if n.startswith("transformer.layers.")})
    if not layers:
        raise RuntimeError("Cannot identify temporal transformer blocks")
    prefix = f"transformer.layers.{layers[-1]}."
    trainable = {n: p for n, p in model.named_parameters() if n.startswith(prefix)}
    for parameter in trainable.values():
        parameter.requires_grad_(True)
    model.train()
    model.transformer.checkpointing = model.depformer.checkpointing = True
    codes = codes.to("cuda")
    reference = asyncio.run(get_conditioning_remote_async(
        "A checked result says bread firms because starch retrogradation reorganizes its structure, not simply drying out.",
        "http://127.0.0.1:18001", timeout=60)).detach()
    times = torch.arange(codes.shape[-1], dtype=torch.float64, device="cuda") * .08
    timeline = reference_schedule(times, [{"ready_at_s": 4., "rows": reference[0]}], model.dim)[None]
    if args.steps:
        model.requires_grad_(False)
        from evals.moshi_rag_adapter_pilot import run_pilot
        report = run_pilot(model, codes, timeline, args.out, args.steps)
        report.update(model=manifest["rag"], load_audit=audit, code_source_sha256=sha256(state_path),
                      scope="optimizer/reload engineering only; reused ten-second sample")
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k:v for k,v in report.items() if k not in ("adapter_keys","trainable_keys","adapter_hashes")}), flush=True)
        return
    conditions = get_condition_tensors("moshi", model, 1, 1., first_speaker="user")
    with no_compile():
        result = forward_with_reference(model, codes, timeline, conditions)
        text = F.cross_entropy(result.text_logits[result.text_mask].float(), codes[:, :1][result.text_mask])
        targets = codes[:, model.audio_offset:model.audio_offset + model.dep_q]
        speech = F.cross_entropy(result.logits[result.mask].float(), targets[result.mask])
        loss = text + speech
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite timed-reference loss")
        loss.backward()
    gradients = {n: p.grad for n, p in trainable.items() if p.grad is not None}
    if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
        raise RuntimeError("Missing or nonfinite timed-reference gradients")
    nonzero = sum(int(g.count_nonzero()) for g in gradients.values())
    if not nonzero:
        raise RuntimeError("Timed-reference gradients are all zero")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"passed": True, "scope": "full pretrained weights, actual ARC HTTP encoder, backward only",
        "code_source_sha256": sha256(state_path), "model": manifest["rag"], "load_audit": audit,
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "codes_shape": list(codes.shape), "reference_shape": list(reference.shape),
        "reference_first_frame": 50, "no_reference_before_ready": bool(timeline[:, :50].eq(0).all()),
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "trainable_prefix": prefix, "nonzero_gradient_elements": nonzero,
        "text_loss": float(text.detach()), "speech_loss": float(speech.detach()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "optimizer_steps": 0, "quality_improvement_tested": False, "saved_candidate": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
