"""Plain words: test whether the untouched base can hear, answer, and speak.

This is a TURN-BASED component diagnostic, never a duplex acceptance test.
Only verified user clips enter the model; the scripted answers are not inputs.
The matching text arm separates audio understanding from answer generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


def select_cases(batch: Path, scenario: str, limit: int) -> list[dict]:
    from data.synthetic.quality import checked_rows

    checked_rows(batch, [{"scenario_id": scenario}], role="native content probe")
    timeline = json.loads((batch / scenario / "timeline.actual.json").read_text())
    cases = []
    for event in timeline["events"]:
        if event["ch"] != "user" or not event.get("text") or event["kind"] == "marker":
            continue
        audio = (batch / event["wav"]).resolve()
        if not audio.is_relative_to(batch.resolve()) or not audio.is_file():
            raise ValueError(f"Invalid verified clip path: {audio}")
        cases.append({"id": f"{scenario}-u{len(cases):02d}",
                      "text": event["text"], "audio": audio,
                      "sha256": hashlib.sha256(audio.read_bytes()).hexdigest()})
        if len(cases) == limit:
            break
    if len(cases) != limit:
        raise ValueError(f"Requested {limit} user clips, found {len(cases)}")
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--scenario", default="dense_dialog-001")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--text-only-output", action="store_true")
    args = ap.parse_args()
    if args.limit < 1:
        ap.error("--limit must be positive")
    cases = select_cases(args.batch, args.scenario, args.limit)
    args.out.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    torch.manual_seed(424242)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; stopping before model load")
    env = {"schema": "dl-native-content-v1", "mode": "turn-based-component-probe",
           "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "model": args.model, "torch": torch.__version__,
           "transformers": transformers.__version__, "machine": platform.machine(),
           "gpu": torch.cuda.get_device_name(), "seed": 424242,
           "audio_output": not args.text_only_output,
           "semantic_verdict": "pending review", "duplex_acceptance": False}
    (args.out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    started = time.monotonic()
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", experts_implementation="batched_mm",
        local_files_only=True).eval()
    proc = Qwen3OmniMoeProcessor.from_pretrained(args.model, local_files_only=True)
    print(f"[native] loaded in {time.monotonic() - started:.1f}s", flush=True)
    env["model_commit"] = getattr(model.config, "_commit_hash", None)
    env["device_map"] = {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()}
    (args.out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    with (args.out / "results.jsonl").open("w") as results, (args.out / "manifest.jsonl").open("w") as manifest:
        for case in cases:
            waveform, sr = sf.read(case["audio"], dtype="float32")
            if waveform.ndim != 1 or sr != 16000:
                raise ValueError(f"Expected verified mono 16k clip, got {waveform.shape}, {sr}")
            for modality in ("text", "audio"):
                sid = f"{case['id']}-{modality}"
                dest = args.out / sid
                dest.mkdir()
                # Same concise instruction, no reference assistant response.
                content = ([{"type": "audio", "audio": str(case["audio"])}]
                           if modality == "audio" else [{"type": "text", "text": case["text"]}])
                messages = [
                    {"role": "system", "content": [{"type": "text", "text":
                        "You are a helpful voice assistant. Answer in English in one short sentence."}]},
                    {"role": "user", "content": content}]
                prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = proc(text=[prompt], audio=[waveform] if modality == "audio" else None,
                              return_tensors="pt", padding=True).to(model.device).to(model.dtype)
                torch.cuda.synchronize()
                started = time.monotonic()
                with torch.inference_mode():
                    ids, audio = model.generate(
                        **inputs, return_audio=not args.text_only_output, speaker="Ethan",
                        thinker_max_new_tokens=96, thinker_do_sample=False,
                        talker_max_new_tokens=512, thinker_return_dict_in_generate=True)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                sequence = ids.sequences if hasattr(ids, "sequences") else ids
                answer_ids = sequence[:, inputs["input_ids"].shape[1]:]
                answer = proc.batch_decode(answer_ids, skip_special_tokens=True)[0]
                rec = {"id": sid, "modality": modality, "question": case["text"],
                       "input_sha256": case["sha256"], "answer": answer,
                       "generation_wall_s": elapsed, "answer_tokens": int(answer_ids.shape[-1]),
                       "hit_text_cap": answer_ids.shape[-1] >= 96,
                       "audio_s": None, "semantic_verdict": "pending review"}
                sf.write(dest / "stem_user.wav", waveform, sr)
                if audio is not None:
                    pcm = audio.detach().float().cpu().numpy().reshape(-1)
                    if not len(pcm) or not np.isfinite(pcm).all():
                        raise ValueError("Empty/nonfinite generated speech")
                    sf.write(dest / "model.wav", pcm, 24000)
                    rec["audio_s"] = len(pcm) / 24000
                    rec["audio_rms"] = float(np.sqrt(np.mean(pcm ** 2)))
                results.write(json.dumps(rec) + "\n")
                results.flush()
                (dest / "result.json").write_text(json.dumps(rec, indent=2) + "\n")
                manifest.write(json.dumps({"scenario_id": sid, "dir": sid,
                                           "scenario_class": "native-component-probe",
                                           "duration_ms": round((rec["audio_s"] or 0) * 1000)}) + "\n")
                manifest.flush()
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
