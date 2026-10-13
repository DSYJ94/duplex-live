"""Paced native MoshiRAG speech with a delayed, fixed reference over actual HTTP.

Called by the audited reference service probe. This tests delivery and speech
use, not learned delegation, microphone latency or real research execution.
"""
import asyncio
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def run_pair(manifest, batch, prefix, single_mode=None, request_aligned=False):
    if single_mode is None:
        # A stream reset in the same process did not reproduce the evaluated
        # prefix, even with longer warmup. Isolate CUDA/compiled/model state;
        # keep the actual HTTP encoder alive in the parent process.
        path = Path(f"{prefix}-model-manifest.json")
        if path.exists():
            raise RuntimeError("Refusing overwrite of speech manifest")
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        for mode in (("delayed-reference",) if request_aligned else ("base", "delayed-reference")):
            command = [sys.executable, "-u", "-m", "evals.moshi_rag_speech_probe",
                            "--manifest", str(path), "--batch", str(batch),
                            "--prefix", str(prefix), "--mode", mode]
            if request_aligned:
                command.append("--request-aligned")
            subprocess.run(command, check=True)
        if request_aligned:
            return  # Standalone task/reference correctness, not a causal pair.
        from evals.moshi_rag_prefix_audit import verify_prefix
        verify_prefix(prefix)
        return
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from moshi.models import loaders, LMGen, LMModel
    from moshi.inference_utils.utils import get_condition_tensors, get_conditioning_remote_async
    from data.synthetic.quality import checked_rows

    torch.manual_seed(424242)
    scenario = "dense_dialog-001"
    seconds = 60
    checked_rows(batch, [{"scenario_id": scenario}], role="MoshiRAG reference replay")
    model_dir = Path(manifest["rag"]["path"])
    checkpoint = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=None, moshi_weights=model_dir / "model.safetensors",
        mimi_weights=model_dir / "tokenizer-e351c8d8-checkpoint125.safetensors",
        tokenizer=model_dir / "tokenizer_spm_32k_3.model", config_path=model_dir / "config.json")
    load_audit = []
    original = LMModel.load_state_dict

    def audited_load(module, *args, **kwargs):
        result = original(module, *args, **kwargs)
        # Reference weights belong to the separately audited HTTP encoder.
        extras = [k for k in result.unexpected_keys if not k.startswith(
            "condition_provider.conditioners.reference_with_time.")]
        if result.missing_keys or extras:
            raise RuntimeError(f"Speech weight mismatch: missing={result.missing_keys}, extra={extras}")
        load_audit.append({"missing": [], "skipped_reference_keys": result.unexpected_keys})
        return result

    LMModel.load_state_dict = audited_load
    try:
        lm = checkpoint.get_moshi(device="cuda", dtype=torch.bfloat16,
                                  skip_conditioners=["reference_with_time"]).eval()
    finally:
        LMModel.load_state_dict = original
    if len(load_audit) != 1:
        raise RuntimeError("Speech loading was not audited")
    lm.requires_grad_(False)
    mimi = checkpoint.get_mimi(device="cuda").eval()
    mimi.requires_grad_(False)
    tokenizer = checkpoint.get_text_tokenizer()
    conditions = get_condition_tensors("moshi", lm, 1, 1.0, first_speaker="user")
    generation = LMGen(lm, condition_tensors=conditions, force_streaming_sum=True,
                       **checkpoint.lm_gen_config)
    frame = int(mimi.sample_rate / mimi.frame_rate)
    pcm, sr = sf.read(batch / scenario / "stem_user.wav", dtype="float32")
    if pcm.ndim != 1 or not np.isfinite(pcm).all():
        raise RuntimeError("Invalid verified user audio")
    if sr != mimi.sample_rate:
        divisor = math.gcd(sr, mimi.sample_rate)
        pcm = resample_poly(pcm, mimi.sample_rate // divisor, sr // divisor).astype(np.float32)
    if len(pcm) < seconds * mimi.sample_rate:
        raise RuntimeError("Input too short")
    pcm = pcm[:seconds * mimi.sample_rate]
    reference = ("Bread goes stale faster in the fridge mainly because refrigeration speeds up starch "
                 "retrogradation. Starch molecules reorganize and firm the crumb. It is not simply drying out.")
    worker_delay = 10.0
    fire_at = 20.0
    fixtures = [
        ("lightning", "You see lightning before hearing thunder because light travels much faster than sound. Both come from the same lightning event.", 0.1),
        ("fridge", "A fridge uses a vapor-compression refrigerant cycle to move heat from inside to outside. The refrigerant evaporates to absorb heat inside, then is compressed and condenses to release heat outside.", 0.1),
        ("bread", reference, 10.0),
        ("noise", "Active noise-cancelling headphones use microphones to measure surrounding noise and produce an opposing sound wave that reduces it through destructive interference.", 0.1),
    ]
    for mode in (single_mode,):
        out = Path(f"{prefix}-{mode}")
        out.mkdir(parents=True, exist_ok=False)
        sf.write(out / "stem_user.wav", pcm, mimi.sample_rate)
        env = {"mode": "native-MoshiRAG-paced-replay", "model": manifest["rag"],
               "source": loaders.__file__, "load_audit": load_audit,
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "seed": 424242, "sampling_seed_reset_after_warmup": True, "warmup_frames": 16,
               "fresh_process": True,
               "cuda_graph_disabled": bool(os.environ.get("NO_CUDA_GRAPH", "")),
               "torch_compile_disabled": bool(os.environ.get("NO_TORCH_COMPILE", "")),
               "reference_mode": mode, "microphone_acceptance": False, "seconds": seconds}
        env["request_aligned"] = request_aligned
        (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
        records, events, output = [], [], np.zeros(0, dtype=np.float32)
        pending = None
        injected = False
        active_reference, active_task = reference, "fixed-bread"
        requested = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            with mimi.streaming(1), generation.streaming(1), torch.inference_mode():
                # Complete lazy compilation and CUDA-graph capture before the
                # evaluated sampling stream. Four frames did not establish a
                # paired prefix; require actual token equality below.
                for _ in range(16):
                    codes = mimi.encode(torch.zeros(1, 1, frame, device="cuda"))
                    tokens = generation.step(codes)
                    if tokens is not None:
                        mimi.decode(tokens[:, 1:])
                torch.cuda.synchronize()
                mimi.reset_streaming()
                generation.reset_streaming()
                torch.manual_seed(424242)
                env["initial_cuda_rng_sha256"] = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
                (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
                start = time.monotonic()

                def worker(reference_text=reference, delay=worker_delay):
                    time.sleep(delay)
                    request_start = time.monotonic() - start
                    tensor = asyncio.run(get_conditioning_remote_async(reference_text, "http://127.0.0.1:18001", timeout=60))
                    return tensor, request_start, time.monotonic() - start

                with (out / "packets.jsonl").open("w") as stream:
                    for index in range(len(pcm) // frame):
                        due = (index + 1) / mimi.frame_rate
                        time.sleep(max(0, start + due - time.monotonic()))
                        before = time.monotonic()
                        if mode == "delayed-reference" and not request_aligned and pending is None and before - start >= fire_at:
                            events.append({"event": "worker_started", "at_s": before - start, "task_id": active_task})
                            pending = pool.submit(worker)
                        if pending is not None and pending.done() and not injected:
                            tensor, requested, ready = pending.result()
                            if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[2] != lm.dim or not torch.isfinite(tensor).all():
                                raise RuntimeError("Invalid delivered reference")
                            generation.update_streaming_sum_tensors([tensor.squeeze(0)])
                            events.append({"event": "reference_injected", "request_at_s": requested,
                                           "ready_at_s": ready, "at_s": time.monotonic() - start,
                                           "chunk": index, "shape": list(tensor.shape), "reference": active_reference,
                                           "task_id": active_task})
                            injected = True
                        codes = mimi.encode(torch.from_numpy(pcm[index * frame:(index + 1) * frame]).cuda()[None, None])
                        input_code_hash = hashlib.sha256(codes.detach().cpu().numpy().tobytes()).hexdigest()
                        rng_hash = (hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
                                    if index < 100 else None)
                        if index == 0:
                            generation.step(codes)
                        tokens = generation.step(codes)
                        text, audio = "", None
                        if tokens is not None:
                            token = int(tokens[0, 0, 0])
                            if token not in (0, 1, 2, 3, 4):
                                text = tokenizer.id_to_piece(token).replace("▁", " ")
                            audio = mimi.decode(tokens[:, 1:]).detach().float().cpu().numpy().reshape(-1)
                        torch.cuda.synchronize()
                        completed = time.monotonic() - start
                        if request_aligned and tokens is not None and int(tokens[0, 0, 0]) == 4:
                            if pending is not None and not injected:
                                raise RuntimeError("Overlapping model requests exceed this ordered fixture's scope")
                            if requested >= len(fixtures):
                                raise RuntimeError("More model requests than reviewed fixture answers")
                            active_task, active_reference, delay = fixtures[requested]
                            requested += 1
                            events.append({"event": "worker_started", "at_s": completed,
                                           "chunk": index, "task_id": active_task, "delay_s": delay,
                                           "trigger": "native rag token"})
                            pending = pool.submit(worker, active_reference, delay)
                            injected = False
                        playback = None
                        if audio is not None:
                            if not np.isfinite(audio).all():
                                raise RuntimeError("Nonfinite generated audio")
                            offset = max(len(output), math.ceil(completed * mimi.sample_rate))
                            playback = offset / mimi.sample_rate
                            output = np.concatenate([np.pad(output, (0, offset - len(output))), audio])
                        token_hash = None if tokens is None else hashlib.sha256(
                            tokens.detach().cpu().numpy().tobytes()).hexdigest()
                        record = {"chunk": index, "input_available_s": due, "input_consumed_s": before - start,
                                  "packet_available_s": completed, "compute_s": completed - (before - start),
                                  "input_backlog_s": before - start - due, "text": text,
                                  "samples": 0 if audio is None else len(audio), "playback_start_s": playback,
                                  "generated_token_sha256": token_hash,
                                  "text_token_id": None if tokens is None else int(tokens[0, 0, 0])}
                        record.update(user_code_sha256=input_code_hash, cuda_rng_before_step_sha256=rng_hash)
                        records.append(record)
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
        if not len(output) or (mode == "delayed-reference" and not any(e["event"] == "reference_injected" for e in events)):
            raise RuntimeError("Missing speech or reference injection")
        sf.write(out / "model.wav", output, mimi.sample_rate)
        waiting = []
        intervals = []
        for event in events:
            if event["event"] == "reference_injected":
                beginning = next(s for s in events if s["event"] == "worker_started" and s["task_id"] == event["task_id"])
                intervals.append((beginning["at_s"], event["ready_at_s"]))
        waiting = [r for r in records if any(a <= r["packet_available_s"] < b for a, b in intervals)]
        report = {"text": "".join(r["text"] for r in records), "task_events": events,
                  "packets_while_worker_waited": len(waiting),
                  "audio_packets_while_worker_waited": sum(r["samples"] > 0 for r in waiting),
                  "text_packets_while_worker_waited": sum(bool(r["text"].strip()) for r in waiting),
                  "compute_p50_s": float(np.median([r["compute_s"] for r in records])),
                  "compute_p95_s": float(np.percentile([r["compute_s"] for r in records], 95)),
                  "max_input_backlog_s": max(r["input_backlog_s"] for r in records),
                  "model_rag_trigger_frames": [r["chunk"] for r in records if r["text_token_id"] == 4],
                  "request_aligned": request_aligned, "fixture_selection": "ordered known questions; no semantic task routing",
                  "learned_delegation_tested": False, "semantic_review": "pending", "human_listening": "pending"}
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "manifest.jsonl").write_text(json.dumps({"scenario_id": scenario, "dir": ".",
            "scenario_class": "native-MoshiRAG-reference-replay", "duration_ms": seconds * 1000}) + "\n")
        print(json.dumps({"mode": mode, **report}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--mode", choices=("base", "delayed-reference"), required=True)
    parser.add_argument("--request-aligned", action="store_true")
    args = parser.parse_args()
    run_pair(json.loads(args.manifest.read_text()), args.batch, args.prefix, args.mode, args.request_aligned)
