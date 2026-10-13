"""Plain words: separate saved-weight errors from native inference-path rounding."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from moshi.models import loaders
    from moshi.utils.compile import no_compile
    from evals.moshi_training_smoke import logits_sample, tensor_hashes

    args.out.mkdir(parents=True, exist_ok=False)
    native = Path(json.loads(args.model_manifest.read_text())["native"]["path"])
    checkpoint = args.run / "train/checkpoints/checkpoint_000004/consolidated"
    reference = torch.load(args.run / "fixed-logits.pt", map_location="cpu", weights_only=True)
    cp = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=None, moshi_weights=native / "model.safetensors",
        mimi_weights=native / "tokenizer-e351c8d8-checkpoint125.safetensors",
        tokenizer=native / "tokenizer_spm_32k_3.model",
        config_path=checkpoint / "config.json", lora_weights=checkpoint / "lora.safetensors")
    model = cp.get_moshi(device="cuda", dtype=torch.bfloat16, fuse_lora=False).eval()
    saved_adapter = load_file(checkpoint / "lora.safetensors")
    loaded_adapter = {n: p.detach().cpu() for n, p in model.named_parameters() if n in saved_adapter}
    if set(loaded_adapter) != set(reference["adapter_keys"]) or any(
            not torch.equal(loaded_adapter[n], p) for n, p in saved_adapter.items()):
        raise RuntimeError("Reloaded adapter differs from saved bytes")
    frozen_mismatches = []
    with safe_open(native / "model.safetensors", framework="pt", device="cpu") as source:
        for name, parameter in model.named_parameters():
            if name in loaded_adapter:
                continue
            base_name = name.replace(".frozen_W.", ".")
            expected = source.get_tensor(base_name).to(parameter.dtype)
            if not torch.equal(parameter.detach().cpu(), expected):
                frozen_mismatches.append(name)
    if frozen_mismatches:
        raise RuntimeError(f"Reloaded base differs: {frozen_mismatches}")
    records = []
    for frozen, checkpointing, compile_enabled in ((False, False, True), (True, False, True),
                                                  (True, True, True), (True, True, False)):
        for name, p in model.named_parameters():
            p.requires_grad = not frozen or name in saved_adapter
        model.transformer.checkpointing = checkpointing
        model.depformer.checkpointing = checkpointing
        if compile_enabled:
            result = logits_sample(model, reference["codes"])
            repeat = logits_sample(model, reference["codes"])
        else:
            with no_compile():
                result = logits_sample(model, reference["codes"])
                repeat = logits_sample(model, reference["codes"])
        rec = {"frozen_base": frozen, "checkpointing": checkpointing, "torch_compile": compile_enabled,
               "error": {k: {"max_abs": float((result[k]-reference[k]).abs().max()),
                              "mean_abs": float((result[k]-reference[k]).abs().mean()),
                              "repeat_max_abs": float((result[k]-repeat[k]).abs().max())} for k in result}}
        records.append(rec)
        print(json.dumps(rec), flush=True)
    report = {"adapter_exact": True, "base_exact": True, "cases": records}
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
