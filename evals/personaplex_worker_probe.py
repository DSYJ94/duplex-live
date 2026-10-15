"""Actual worker-result replay through the accepted PersonaPlex voice.

One scheduled diagnostic dispatch, causal measured transcript oracle. PersonaPlex
does NOT yet choose delegation. Real tools/API and frozen ARC produce the result;
only its completed rows may enter the trained bridge. No fixed answer bank.
"""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "voice", "voice-provenance", "data", "bridge", "arc-source", "manifest", "arc-tokenizer", "out"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--scenario", default="recovery-validation-000")
    args = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import moshi
    from moshi import offline
    from safetensors.torch import load_file
    from evals.recovery_data import validate_conversion
    from evals.recovery_context import causal_dialogue
    from evals.recovery_worker import RecordWorld, run_worker
    from evals.personaplex_reference import ReferenceBridge
    from evals.personaplex_arc import load_encoder
    from evals.personaplex_pacing import paced_export
    if Path(moshi.__file__).resolve() != (args.source / "moshi" / "moshi" / "__init__.py").resolve():
        raise RuntimeError("Wrong foreground source")
    if (subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
            != "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
            or subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain"], text=True)):
        raise RuntimeError("Foreground source changed")
    model_hashes = {"model.safetensors": "db1290db583cdaa6cb4de444ed279e0b586ca2a372b41434b07a7461c8c0e2f4",
        "tokenizer-e351c8d8-checkpoint125.safetensors": "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
        "tokenizer_spm_32k_3.model": "78d4336533ddc26f9acf7250d7fb83492152196c6ea4212c841df76933f18d2d"}
    if any(sha256(args.model / name) != digest for name, digest in model_hashes.items()):
        raise RuntimeError("Accepted foreground assets changed")
    conversion = validate_conversion(args.data)
    row = next(r for r in conversion["accepted"] if r["scenario_id"] == args.scenario)
    if row["split"] != "validation" or row["kind"] != "private":
        raise RuntimeError("This diagnostic requires an approved private validation scene")
    sidecar = json.loads(Path(row["path"]).with_suffix(".json").read_text())
    metadata = sidecar["timeline"]["meta"]
    voice_receipt = json.loads(args.voice_provenance.read_text())
    if (voice_receipt.get("schema") != "dl-voice-codec-probe-v1"
            or voice_receipt.get("source_model") != "gpt-live-1"
            or voice_receipt.get("reference_sha256") != sha256(args.voice)):
        raise RuntimeError("Accepted synthetic voice provenance changed")
    trained = json.loads((args.bridge / "report.json").read_text())
    reload = json.loads((args.bridge / "reload-report.json").read_text())
    if not trained.get("passed") or not reload.get("passed") or trained["bridge_sha256"] != sha256(args.bridge / "bridge.safetensors"):
        raise RuntimeError("Unverified bridge checkpoint")
    if trained["voice_sha256"] != sha256(args.voice):
        raise RuntimeError("Bridge used a different voice prompt")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "environment.json").write_text(json.dumps({"repo_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "model_hashes": model_hashes,
        "voice_sha256": sha256(args.voice), "bridge_sha256": trained["bridge_sha256"],
        "torch": torch.__version__, "native_routing": False, "fresh_holdout": False,
        "arc_manifest_sha256": sha256(args.manifest), "context": "causal measured-word oracle"}, indent=2)+"\n")
    os.environ["TRITON_LIBCUDA_PATH"] = subprocess.check_output([
        sys.executable, "-m", "evals.cuda_linker_path", "--out", str(args.out / "driver")], text=True).strip()
    encoder = load_encoder(args.arc_source, args.manifest, args.arc_tokenizer, args.out)
    pcm, rate = sf.read(row["path"], dtype="float32", always_2d=True)
    if rate != 24000 or pcm.shape[1] != 2 or not np.isfinite(pcm).all():
        raise RuntimeError("Invalid native approved user audio")
    pcm = np.concatenate([pcm[:, 1], np.zeros(20*rate, dtype=np.float32)])
    sf.write(args.out / "user_in.wav", pcm, rate)
    world = RecordWorld({metadata["project"]: {metadata["field"]: metadata["phrase"]}})
    user_event = next(e for e in sidecar["timeline"]["events"] if e["ch"] == "user" and e["kind"] == "utterance")
    trigger = user_event["actual_end_ms"]/1000 + .16
    events, holder, pending, future = [], {}, [], None
    original_loader = offline.loaders.get_moshi_lm

    def load_foreground(*pos, **kw):
        lm = original_loader(*pos, **kw).eval().requires_grad_(False)
        bridge = ReferenceBridge(4096, lm.dim, rank=64, device="cuda", dtype=torch.float32)
        bridge.load_state_dict(load_file(args.bridge / "bridge.safetensors", device="cuda"), strict=True)
        bridge.eval().requires_grad_(False)
        holder.update(bridge=bridge, hook=lm.transformer.register_forward_pre_hook(bridge.temporal_input))
        return lm

    def work(context, revision):
        result = run_worker(context, world, "diagnostic-request", expected_revision=revision)
        if not result["receipts"] or not any(r.get("tool") == "lookup_record" for r in result["receipts"]):
            raise RuntimeError("Scheduled private diagnostic did not execute a real lookup")
        with torch.inference_mode():
            rows = encoder.encode(result["content"]).detach().float().cpu()
        if rows.ndim != 3 or rows.shape[0] != 1 or rows.shape[-1] != 4096 or not torch.isfinite(rows).all():
            raise RuntimeError("Invalid actual worker reference rows")
        return result, rows[0], time.monotonic()

    started = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        def before_frame(frame, prior):
            nonlocal started, future
            if started is None:
                started = time.monotonic()-frame["input_consumed_s"]
            bridge = holder["bridge"]
            bridge.clear()
            due = frame["input_available_s"]
            if future is None and due >= trigger:
                context = causal_dialogue(sidecar, due, [], frame["input_consumed_s"])
                revision = world.reserve("diagnostic-request", context=context)
                future = pool.submit(work, context, revision)
                events.append({"event": "worker_started", "at_s": time.monotonic()-started,
                    "input_available_s": due, "context": context,
                    "trigger": "scheduled diagnostic after measured request; not native routing"})
            if future is not None and future.done() and not holder.get("delivered"):
                result, rows, ready = future.result()
                pending.extend(rows)
                holder["delivered"] = True
                torch.save(rows, args.out / "actual-worker-reference.pt")
                events.append({"event": "worker_completed", "ready_at_s": ready-started,
                    "observed_at_s": time.monotonic()-started, "rows": len(rows), "result": result})
            if pending:
                bridge.set_row(pending.pop(0))
                frame["reference_l1"] = float(bridge.active.abs().sum())
                events.append({"event": "reference_applied", "frame": frame["frame"], "at_s": time.monotonic()-started})
            else:
                frame["reference_l1"] = 0.

        offline.loaders.get_moshi_lm = load_foreground
        try:
            with paced_export(offline, args.out, torch.cuda.synchronize, before_frame) as timing:
                offline.run_inference(input_wav=str(args.out / "user_in.wav"), output_wav=str(args.out / "model_out.wav"),
                    output_text=str(args.out / "output.json"),
                    text_prompt="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
                    voice_prompt_path=str(args.voice), tokenizer_path=str(args.model / "tokenizer_spm_32k_3.model"),
                    moshi_weight=str(args.model / "model.safetensors"),
                    mimi_weight=str(args.model / "tokenizer-e351c8d8-checkpoint125.safetensors"),
                    hf_repo="nvidia/personaplex-7b-v1", device="cuda", seed=42424242,
                    temp_audio=.8, temp_text=.7, topk_audio=250, topk_text=25,
                    greedy=False, save_voice_prompt_embeddings=False)
        finally:
            offline.loaders.get_moshi_lm = original_loader
            if "hook" in holder:
                holder["hook"].remove()
            (args.out / "events.json").write_text(json.dumps(events, indent=2)+"\n")
    if not holder.get("delivered") or not any(e["event"] == "reference_applied" for e in events):
        raise RuntimeError("No completed actual worker result entered the native foreground")
    tokens = json.loads((args.out / "output.json").read_text())
    report = {"schema": "dl-personaplex-worker-probe-v1", "scenario": args.scenario,
        "text": "".join(t for t in tokens if t not in ("EPAD", "PAD", "BOS", "EOS")),
        "timing": timing, "bridge_sha256": trained["bridge_sha256"], "voice_sha256": sha256(args.voice),
        "native_routing": False, "context": "causal measured transcript oracle, not live ASR",
        "worker": "actual API/tool receipts and frozen in-process ARC; no ordered answer bank",
        "events": events, "complete_task_success": "pending grading", "microphone_acceptance": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "events"}), flush=True)


if __name__ == "__main__":
    main()
