"""Check pinned ARC weights and the actual HTTP reference-encoding path."""
import argparse
import asyncio
import json
import subprocess
import threading
import time
from pathlib import Path

from evals.moshi_model_download import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--speech-batch", type=Path)
    parser.add_argument("--speech-prefix", type=Path)
    parser.add_argument("--request-aligned", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import torch
    import uvicorn
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from moshi import server_conditioner as upstream
    from moshi.inference_utils.utils import get_conditioning_remote_async

    manifest = json.loads(args.manifest.read_text())
    # Recheck the exact artifacts used; no mutable upstream weight download.
    for label in ("rag", "arc"):
        entry = manifest[label]
        for name, record in entry["files"].items():
            if name not in ("config.json", "model.safetensors"):
                continue
            path = Path(entry["path"]) / name
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise RuntimeError(f"Changed pinned artifact: {label}/{name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, local_files_only=True)
    tokenizer_record = {"repo": "meta-llama/Llama-3.2-3B-Instruct",
                        "revision": args.tokenizer.name,
                        "path": str(args.tokenizer), "available": True,
                        "vocab_size": tokenizer.vocab_size,
                        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                                  for p in args.tokenizer.iterdir() if p.is_file()}}
    (args.out / "tokenizer-manifest.json").write_text(json.dumps(tokenizer_record, indent=2) + "\n")
    config = json.loads((Path(manifest["rag"]["path"]) / "config.json").read_text())
    ref_config = config["conditioners"]["reference_with_time"]["multi_arc_encoder"]
    ref_config.update(tokenizer_name=str(args.tokenizer), hf_repo=None)
    config_path = args.out / "local-config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    report = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "upstream": upstream.__file__, "tokenizer": tokenizer_record,
              "scope": "reference encoder and HTTP transport; no speech or task-dispatch acceptance"}

    class AuditedService(upstream.EncoderService):
        def _load_conditioner_weights_from_moshi_checkpoint(self, moshi_weight):
            prefix = "condition_provider.conditioners.reference_with_time."
            with safe_open(moshi_weight, framework="pt", device="cpu") as source:
                state = {k.removeprefix(prefix): source.get_tensor(k)
                         for k in source.keys() if k.startswith(prefix)}
            speech_keys = set(state)
            with safe_open(str(Path(manifest["arc"]["path"]) / "model.safetensors"),
                           framework="pt", device="cpu") as source:
                arc_keys = set(source.keys())
                state.update({k: source.get_tensor(k) for k in source.keys()})
            module = self.condition_provider.conditioners[self.conditioner_name]
            # Same precedence as upstream: speech conditioner first, ARC second.
            # Unlike its strict=False calls, require complete final coverage.
            expected = set(module.state_dict())
            # These exact released ARC tensors are absent from, and never read
            # by, the pinned inference transformer. Upstream ignores them via
            # strict=False. Document this exception and reject every other key.
            unused = {"embedder.cont_tok.0.weight", "embedder.rec_tok.0.weight"}
            if set(state) - expected != unused:
                raise RuntimeError(f"Unexpected ARC extras: {sorted(set(state)-expected)}")
            unused_shapes = {k: list(state.pop(k).shape) for k in sorted(unused)}
            if set(state) != expected:
                raise RuntimeError(f"Incomplete merged encoder: missing={sorted(expected-set(state))}, "
                                   f"unexpected={sorted(set(state)-expected)}")
            module.load_state_dict(state, strict=True)
            module.requires_grad_(False)
            module.eval()
            if not all(torch.isfinite(p).all().item() for p in module.parameters()):
                raise RuntimeError("Nonfinite encoder weight")
            report["weights"] = {"speech_keys": len(speech_keys), "arc_keys": len(arc_keys),
                                 "overlap": sorted(speech_keys & arc_keys),
                                 "unused_checkpoint_tensors": unused_shapes,
                                 "complete_keys": len(expected), "strict_load": True}

    start = time.monotonic()
    with torch.inference_mode():
        service = AuditedService(str(config_path),
                                 str(Path(manifest["rag"]["path"]) / "model.safetensors"),
                                 "reference_with_time", "cuda:0")
        cold_start = time.monotonic()
        warm = service.encode("Reference encoder warmup.")
        torch.cuda.synchronize()
        report["load_seconds"] = cold_start - start
        report["cold_encode_seconds"] = time.monotonic() - cold_start
        if not torch.isfinite(warm).all():
            raise RuntimeError("Nonfinite warmup")
    upstream._service = service
    # Actual localhost TCP transport and upstream endpoint/consumer, not a stub.
    server = uvicorn.Server(uvicorn.Config(upstream.app, host="127.0.0.1", port=18001,
                                          log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("Encoder HTTP startup failed")
        time.sleep(0.05)
    texts = ["", "The access code is cobalt.",
             "Bread firms mainly because starch retrogradation changes its structure, not simply because it dries out.",
             "The access code is cobalt.",
             "A background worker checked the report. Its result should update the current answer while conversation continues. " * 20]

    async def requests():
        outputs, records = [], []
        for index, text in enumerate(texts):
            request_start = time.monotonic()
            tensor = await get_conditioning_remote_async(text, "http://127.0.0.1:18001", timeout=60)
            elapsed = time.monotonic() - request_start
            if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] == 0 or tensor.shape[2] != config["dim"]:
                raise RuntimeError(f"Invalid embedding shape: {tuple(tensor.shape)}")
            # The official empty-reference branch deliberately emits zero
            # vectors. Require a signal only when actual text was supplied.
            if not torch.isfinite(tensor).all() or (text and tensor.abs().sum() == 0):
                raise RuntimeError(f"Invalid encoder values for request {index}")
            records.append({"index": index, "text": text, "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
                            "shape": list(tensor.shape), "dtype": str(tensor.dtype), "http_seconds": elapsed})
            outputs.append(tensor)
        error = float((outputs[1].float() - outputs[3].float()).abs().max())
        if error != 0.0:
            raise RuntimeError(f"Repeat reference changed: {error}")
        if outputs[0].shape == outputs[1].shape and torch.equal(outputs[0], outputs[1]):
            raise RuntimeError("Nonempty text equals empty condition")
        report.update(requests=records, repeat_max_abs_error=error, passed=True)

    try:
        asyncio.run(requests())
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k not in ("tokenizer", "weights")}), flush=True)
        if args.speech_batch:
            if args.speech_prefix is None:
                raise RuntimeError("Speech replay needs an output prefix")
            from evals.moshi_rag_speech_probe import run_pair
            run_pair(manifest, args.speech_batch, args.speech_prefix, request_aligned=args.request_aligned)
    finally:
        server.should_exit = True
        thread.join(timeout=10)


if __name__ == "__main__":
    main()
