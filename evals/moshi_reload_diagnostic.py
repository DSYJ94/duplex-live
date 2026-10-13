"""Plain words: separate saved-weight errors from native inference-path rounding."""
import argparse
import json
import re
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
    from evals.moshi_training_smoke import logits_sample

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
        source_keys = set(source.keys())
        for name, parameter in model.named_parameters():
            if name in loaded_adapter:
                continue
            base_name = name.replace(".frozen_W.", ".")
            if base_name in source_keys:
                expected = source.get_tensor(base_name)
            else:
                # Apply the pinned official attention load hook's legacy
                # combined-projection -> per-step projection transformation.
                match = re.fullmatch(r"(.*\.)(in_projs|out_projs)\.(\d+)\.weight", base_name)
                if match is None:
                    raise RuntimeError(f"Unknown base weight mapping: {base_name}")
                prefix, kind, index = match.groups()
                legacy_names = ("in_proj_weight", "in_proj.weight") if kind == "in_projs" else ("out_proj.weight",)
                legacy = next((prefix+n for n in legacy_names if prefix+n in source_keys), None)
                if legacy is None:
                    raise RuntimeError(f"Missing legacy projection: {base_name}")
                value = source.get_tensor(legacy)
                module = model.get_submodule(prefix[:-1])
                expected = value.view(module.mult, -1, *value.shape[1:])[int(index)]
            expected = expected.to(parameter.dtype)
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
