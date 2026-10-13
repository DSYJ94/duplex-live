"""Plain words: audit the official native Moshi trainer and saved adapter.

This is a single-GPU engineering smoke, not retrieval-model training. Runtime
guards expose missing weights, bad losses/gradients and serialization errors.
The unchanged pinned trainer retains its own mixed-precision optimizer logic.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_data(data):
    """Recheck source approvals AND exact converted artifacts at training startup."""
    from data.synthetic.quality import checked_rows
    summary = json.loads((data / "summary.json").read_text())
    ingestion = json.loads((data / "ingestion.json").read_text())
    if not summary.get("verified") or len(ingestion["accepted"]) != 20:
        raise RuntimeError("Native smoke requires twenty approved aligned examples")
    for name, digest in summary["manifest_sha256"].items():
        if file_hash(data / name) != digest:
            raise RuntimeError("Converted manifest changed after verification")
    by_batch = {}
    for row in ingestion["accepted"]:
        by_batch.setdefault(row["source_batch"], []).append(row["source_row"])
        source = Path(row["source_batch"]) / row["source_row"]["dir"]
        checks = {Path(row["path"]): row["wav_sha256"],
                  Path(row["path"]).with_suffix(".json"): row["sidecar_sha256"],
                  source / "timeline.actual.json": row["source_timeline_sha256"],
                  source / "stem_assistant.wav": row["source_assistant_sha256"],
                  source / "stem_user.wav": row["source_user_sha256"],
                  Path(row["source_batch"]) / "quality.jsonl": row["source_quality_sha256"]}
        if any(file_hash(path) != digest for path, digest in checks.items()):
            raise RuntimeError(f"Training artifacts changed: {row['scenario_id']}")
    for batch, rows in by_batch.items():
        checked_rows(Path(batch), rows, role="native Moshi training smoke")


def tensor_hashes(parameters):
    import torch
    return {name: hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, p in parameters}


def logits_sample(model, codes):
    import torch
    with torch.inference_mode():
        result = model(codes=codes.to(model.device))
    return {"text": result.text_logits[..., :64, :64].float().cpu(),
            "audio": result.logits[..., :64, :64].float().cpu()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trainer", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    validate_data(args.data)
    import torch
    import yaml
    from safetensors.torch import load_file
    from moshi.models import loaders
    from moshi.models.lm import LMModel

    if "moshi-rag" in loaders.__file__:
        raise RuntimeError("Native pilot imported retrieval fork")
    native = json.loads(args.model_manifest.read_text())["native"]
    model_dir = Path(native["path"])
    def info(config, adapter=None):
        return loaders.CheckpointInfo.from_hf_repo(
            hf_repo=None, moshi_weights=model_dir / "model.safetensors",
            mimi_weights=model_dir / "tokenizer-e351c8d8-checkpoint125.safetensors",
            tokenizer=model_dir / "tokenizer_spm_32k_3.model", config_path=config,
            lora_weights=adapter)

    if args.reload:
        checkpoint = args.out / "train/checkpoints/checkpoint_000004/consolidated"
        state = torch.load(args.out / "fixed-logits.pt", map_location="cpu", weights_only=True)
        model = info(checkpoint / "config.json", checkpoint / "lora.safetensors").get_moshi(
            device="cuda", dtype=torch.bfloat16, fuse_lora=False).eval()
        expected_keys = set(state["adapter_keys"])
        actual_keys = set(load_file(checkpoint / "lora.safetensors"))
        if actual_keys != expected_keys:
            raise RuntimeError("Saved adapter key set differs from training parameters")
        sample = logits_sample(model, state["codes"])
        errors = {k: float((sample[k]-state[k]).abs().max().item()) for k in sample}
        if any(not torch.isfinite(sample[k]).all() or errors[k] > .05 for k in sample):
            raise RuntimeError(f"Fresh-process reload differs: {errors}")
        report = {"reload_ok": True, "max_abs_logit_error": errors,
                  "adapter_keys": len(expected_keys), "module": loaders.__file__}
        (args.out / "reload-report.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
        return

    args.out.mkdir(parents=True, exist_ok=False)
    base_config = args.out / "base-config.json"
    base_config.write_text(json.dumps(copy.deepcopy(loaders._lm_kwargs), indent=2)+"\n")
    config = {"data": {"train_data": str(args.data / "train16.jsonl"),
                       "eval_data": str(args.data / "eval4.jsonl"), "shuffle": False},
              "moshi_paths": {"hf_repo_id": None, "moshi_path": str(model_dir / "model.safetensors"),
                              "mimi_path": str(model_dir / "tokenizer-e351c8d8-checkpoint125.safetensors"),
                              "tokenizer_path": str(model_dir / "tokenizer_spm_32k_3.model"),
                              "config_path": str(base_config)},
              "full_finetuning": False, "lora": {"enable": True, "rank": 8, "scaling": 2., "ft_embed": False},
              "duration_sec": 180, "batch_size": 1, "num_microbatches": 4, "max_steps": 4,
              "gradient_checkpointing": True, "param_dtype": "bfloat16",
              "first_codebook_weight_multiplier": 100., "text_padding_weight": .5,
              "optim": {"lr": 2e-6, "weight_decay": .1, "pct_start": .5},
              "seed": 0, "log_freq": 1, "do_eval": True, "eval_freq": 0,
              "do_ckpt": True, "ckpt_freq": 0, "save_adapters": True, "num_ckpt_keep": 1,
              "run_dir": str(args.out / "train"), "overwrite_run_dir": False}
    config_path = args.out / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config))
    sys.path.insert(0, str(args.trainer))
    import train as official
    from finetune.data.interleaver import InterleavedTokenizer
    audit = {"schema": "dl-native-moshi-training-smoke-v1", "native_model": native,
             "torch": torch.__version__, "module": loaders.__file__, "trainer": str(args.trainer),
             "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
             "optimizer_steps": [], "sample_paths": [], "independent_generalization": False}
    captured = {}
    original_load = LMModel.load_state_dict
    def strict_base_load(model, state, *pos, **kw):
        result = original_load(model, state, *pos, **kw)
        allowed = {n for n, _ in model.named_parameters() if ".lora_A." in n or ".lora_B." in n}
        if set(result.missing_keys) != allowed or result.unexpected_keys:
            raise RuntimeError(f"Unexpected base loading keys: {result}")
        audit["base_load_missing_adapter_keys"] = sorted(allowed)
        return result
    LMModel.load_state_dict = strict_base_load
    original_get_model = official.get_fsdp_model
    def audited_get_model(*pos, **kw):
        model = original_get_model(*pos, **kw)
        if any(p.is_meta or not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Meta or nonfinite model parameter")
        captured["model"] = model
        captured["initial_adapter"] = tensor_hashes((n, p) for n, p in model.named_parameters() if p.requires_grad)
        captured["frozen"] = tensor_hashes((n, p) for n, p in model.named_parameters() if not p.requires_grad)
        audit["trainable"] = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
        audit["trainable_count"] = sum(audit["trainable"].values())
        return model
    official.get_fsdp_model = audited_get_model
    original_tokenize = InterleavedTokenizer.__call__
    def traced_tokenize(self, wav, start_sec, path):
        audit["sample_paths"].append({"path": path, "start_s": start_sec})
        sample = original_tokenize(self, wav, start_sec, path)
        if "codes" not in captured:
            captured["codes"] = sample.codes[..., :128].detach().cpu()
        return sample
    InterleavedTokenizer.__call__ = traced_tokenize
    original_loss = official.compute_loss_with_mask
    def checked_loss(logits, target, target_mask, **kw):
        if not target_mask.any():
            raise RuntimeError("Empty loss mask")
        loss = original_loss(logits, target, target_mask, **kw)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite native speech loss")
        return loss
    official.compute_loss_with_mask = checked_loss
    original_step = torch.optim.AdamW.step
    def checked_step(optimizer, *pos, **kw):
        gradients = [p.grad for group in optimizer.param_groups for p in group["params"]
                     if p.requires_grad and p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("Missing or nonfinite adapter gradients")
        nonzero = sum(int(g.count_nonzero().item()) for g in gradients)
        if not nonzero:
            raise RuntimeError("All adapter gradients are zero")
        audit["optimizer_steps"].append({"nonzero_gradient_elements": nonzero})
        return original_step(optimizer, *pos, **kw)
    torch.optim.AdamW.step = checked_step
    official.train(str(config_path))
    LMModel.load_state_dict = original_load
    model = captured["model"].eval()
    final_frozen = tensor_hashes((n, p) for n, p in model.named_parameters() if not p.requires_grad)
    if captured["frozen"] != final_frozen:
        raise RuntimeError("Frozen base parameters changed")
    final_adapter = tensor_hashes((n, p) for n, p in model.named_parameters() if p.requires_grad)
    audit["changed_adapter_tensors"] = sum(v != captured["initial_adapter"][n] for n, v in final_adapter.items())
    if not audit["changed_adapter_tensors"] or len(audit["optimizer_steps"]) != 4:
        raise RuntimeError("Adapter update count invalid")
    logits = logits_sample(model, captured["codes"])
    torch.save({"codes": captured["codes"], "adapter_keys": sorted(audit["trainable"]), **logits},
               args.out / "fixed-logits.pt")
    audit["frozen_base_unchanged"] = True
    audit["peak_memory_bytes"] = torch.cuda.max_memory_allocated()
    (args.out / "training-report.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({k: v for k, v in audit.items() if k not in ("native_model", "trainable", "base_load_missing_adapter_keys")}), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
