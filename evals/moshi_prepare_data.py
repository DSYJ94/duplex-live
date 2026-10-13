"""Plain words: convert verified conversation stems into audited native Moshi data.

Only complete dialog utterances enter this engineering smoke. Word intervals
come from ASR alignment on the actual assistant stem, never fabricated ends.
The resulting folds are diagnostics, not a fresh generalization benchmark.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-count", type=int, default=16)
    ap.add_argument("--eval-count", type=int, default=4)
    args = ap.parse_args()
    if args.train_count < 1 or args.eval_count < 1:
        ap.error("Positive train/eval counts required")
    from data.synthetic.quality import checked_rows
    from data.synthetic.filter import load_asr, norm_words, wer
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    args.batch = args.batch.resolve()
    args.out = args.out.resolve()
    rows = [json.loads(line) for line in (args.batch / "train_manifest.jsonl").read_text().splitlines()]
    candidates = sorted((r for r in rows if r.get("scenario_class") == "dense_dialog"
                         and r.get("duration_ms", 0) <= 100000), key=lambda r: r["scenario_id"])
    checked_rows(args.batch, candidates, role="native Moshi data conversion")
    args.out.mkdir(parents=True, exist_ok=False)
    asr = load_asr("openai/whisper-large-v3-turbo")
    accepted = []
    rejects = []
    for row in candidates:
        sid = row["scenario_id"]
        source = args.batch / row["dir"]
        timeline = json.loads((source / "timeline.actual.json").read_text())
        try:
            if any(e["ch"] not in ("assistant", "user") or e["kind"] != "utterance"
                   for e in timeline["events"]):
                raise ValueError("Smoke requires two speakers and complete utterances")
            assistant, sr = sf.read(source / "stem_assistant.wav", dtype="float32")
            user, user_sr = sf.read(source / "stem_user.wav", dtype="float32")
            if sr != 24000 or sr != user_sr or assistant.ndim != 1 or user.shape != assistant.shape:
                raise ValueError("Misaligned or nonstandard stereo stems")
            stereo = np.stack([assistant, user], axis=1)
            if not np.isfinite(stereo).all() or np.max(np.abs(stereo)) > 1:
                raise ValueError("Nonfinite or clipped stems")
            duration = len(assistant) / sr
            if duration > 100 or abs(duration - row["duration_ms"] / 1000) > .08:
                raise ValueError("Decoded duration differs from approved manifest")
            words = []
            audits = []
            for event in timeline["events"]:
                if event["ch"] != "assistant":
                    continue
                start, end = event["actual_start_ms"] / 1000, event["actual_end_ms"] / 1000
                clip = assistant[round(start * sr):round(end * sr)]
                gcd = math.gcd(sr, 16000)
                pcm = resample_poly(clip, 16000 // gcd, sr // gcd).astype(np.float32)
                aligned = asr({"array": pcm, "sampling_rate": 16000}, return_timestamps="word",
                              generate_kwargs={"language": "english", "task": "transcribe"})
                error_rate = wer(norm_words(event["text"]), norm_words(aligned["text"]))
                if error_rate > .08:
                    raise ValueError(f"Word-alignment transcript fails: WER={error_rate}")
                chunks = aligned.get("chunks", [])
                if not chunks:
                    raise ValueError("No measured word intervals")
                local = []
                for word in chunks:
                    a, b = word["timestamp"]
                    if a is None or b is None or not 0 <= a < b <= end-start+.08:
                        raise ValueError("Invalid measured word interval")
                    local.append([word["text"].strip(), [start+a, min(start+b, end)], "SPEAKER_MAIN"])
                words.extend(local)
                audits.append({"start_s": start, "end_s": end, "expected": event["text"],
                               "asr": aligned["text"], "wer": error_rate, "n_words": len(local)})
            if not words or any(a[1][1] > b[1][0]+.02 for a, b in zip(words, words[1:])):
                raise ValueError("Empty, unsorted or overlapping assistant word intervals")
            target = args.out / sid
            target.mkdir()
            wav = target / "conversation.wav"
            sidecar = target / "conversation.json"
            sf.write(wav, stereo, sr, subtype="FLOAT")
            sidecar.write_text(json.dumps({"alignments": words}, indent=2) + "\n")
            accepted.append({"path": str(wav), "duration": duration, "source_batch": str(args.batch),
                             "scenario_id": sid, "source_row": row, "verified": True,
                             "source_quality_sha256": sha256(args.batch / "quality.jsonl"),
                             "source_timeline_sha256": sha256(source / "timeline.actual.json"),
                             "source_assistant_sha256": sha256(source / "stem_assistant.wav"),
                             "source_user_sha256": sha256(source / "stem_user.wav"),
                             "wav_sha256": sha256(wav), "sidecar_sha256": sha256(sidecar),
                             "alignment_method": "Whisper word timestamps on approved assistant stem crops",
                             "utterance_audits": audits})
            print(json.dumps({"event": "accepted", "id": sid, "n_words": len(words)}), flush=True)
        except ValueError as error:
            rejects.append({"scenario_id": sid, "reason": str(error)})
            print(json.dumps({"event": "rejected", **rejects[-1]}), flush=True)
        (args.out / "ingestion.json").write_text(json.dumps(
            {"schema": "dl-moshi-ingestion-v1", "accepted": accepted, "rejected": rejects}, indent=2)+"\n")
        if len(accepted) == args.train_count + args.eval_count:
            break
    if len(accepted) != args.train_count + args.eval_count:
        raise RuntimeError("Not enough approved aligned conversations; training must not start")
    for label, examples in (("train16", accepted[:args.train_count]), ("eval4", accepted[args.train_count:])):
        (args.out / f"{label}.jsonl").write_text("".join(
            json.dumps({"path": r["path"], "duration": r["duration"]})+"\n" for r in examples))
    (args.out / "summary.json").write_text(json.dumps({
        "schema": "dl-moshi-ingestion-v1", "verified": True,
        "train_count": args.train_count, "eval_count": args.eval_count,
        "asr_revision": getattr(asr.model.config, "_commit_hash", None),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "independent_generalization_split": False,
        "human_listening": "pending", "manifest_sha256": {
            name: sha256(args.out/name) for name in ("train16.jsonl", "eval4.jsonl")}}, indent=2)+"\n")


if __name__ == "__main__":
    main()
