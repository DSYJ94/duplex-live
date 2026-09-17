"""Model-based quality filter for synthetic-factory outputs (Issue #3).

Two passes over a rendered batch (render.py output dir):

1. Rule checks (no model): clipping, same-channel event collisions,
   barge-in cut placement, dead-air fraction.
2. Judge pass (open ASR — ADR-0002): per-channel stem round-trip. Each
   stem is transcribed and compared against the concatenation of that
   channel's intended texts (truncated events contribute their estimated
   spoken prefix). WER thresholds gate the scenario.

Outputs (written next to manifest.jsonl):

  filter_report.jsonl    one line per scenario: checks, per-channel WER, verdict
  train_manifest.jsonl   manifest rows that passed every hard check

Usage (Isambard, GH200, pylibs overlay — see synth-filter.slurm):

  python -m data.synthetic.filter --audio runs/synth/audio
  python -m data.synthetic.filter --audio runs/synth/audio --no-model   # rules only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

GRID_MS = 80
MS_PER_WORD_EST = 320  # TTS rate estimate for truncation-aware ref text
SHORT_REF_WORDS = 4     # <= this many words: duration gate instead of WER
SHORT_MAX_S = 4.0       # "mm-hm" should never run longer than this


# ---------------------------------------------------------------- text utils

def norm_words(text: str) -> list[str]:
    """Lowercase alphanumeric tokens (ASR-hypothesis normalization)."""
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    """Word error rate (edit distance / ref length; 0.0 for empty ref+hyp)."""
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    # Levenshtein over tokens
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1,          # deletion
                         cur[j - 1] + 1,       # insertion
                         prev[j - 1] + (r != h))  # substitution
        prev = cur
    return prev[-1] / len(ref)


# ---------------------------------------------------------------- rule pass

def is_truncated(ev) -> bool:
    return bool(ev.note) and "truncated at barge-in" in ev.note


def rule_checks(tl, sc_dir: Path) -> tuple[list[str], dict]:
    """Hard checks return issues (empty = pass); soft stats go in info."""
    import soundfile as sf

    issues: list[str] = []
    info: dict = {"silence_frac": None, "peak": None}

    speech = sorted([e for e in tl.events if e.text and e.kind != "marker"],
                    key=lambda e: e.actual_start_ms or 0)

    # same-channel collisions: two speech events on one channel overlapping in
    # actual time means the stem summed them -> garbled audio
    by_ch: dict[str, list] = {}
    for ev in speech:
        by_ch.setdefault(ev.ch, []).append(ev)
    for ch, evs in by_ch.items():
        for a, b in zip(evs, evs[1:]):
            if a.actual_end_ms and b.actual_start_ms is not None:
                ov = a.actual_end_ms - b.actual_start_ms
                if ov > GRID_MS:  # tolerate a single grid step of tail bleed
                    issues.append(
                        f"same_channel_collision: {ch} events at "
                        f"{a.actual_start_ms}/{b.actual_start_ms}ms overlap {ov}ms")

    # barge-in cut placement: truncated event must end at (quantized) barge
    # instant and its stem snippet must match the kept duration
    for ev in speech:
        if is_truncated(ev):
            m = re.search(r"truncated at barge-in (\d+)ms", ev.note)
            if not m:
                continue
            cut = int(m.group(1))
            want = round(cut / GRID_MS) * GRID_MS
            if ev.actual_end_ms != want:
                issues.append(f"cut_misplaced: {ev.ch} ends {ev.actual_end_ms}ms, "
                              f"expected {want}ms")
            _wav, sr = sf.read(utt_for(tl, sc_dir, ev), dtype="float32")
            kept_ms = round(len(_wav) / sr * 1000)
            if abs(kept_ms - (ev.actual_end_ms - ev.actual_start_ms)) > 3 * GRID_MS:
                issues.append(f"cut_audio_mismatch: {ev.ch} stem {kept_ms}ms vs "
                              f"timeline {ev.actual_end_ms - ev.actual_start_ms}ms")

    # mixed track: clipping + dead air (informational)
    mixed, sr = sf.read(sc_dir / "mixed.wav", dtype="float32")
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    info["peak"] = round(peak, 3)
    if peak > 0.995:
        issues.append(f"clipping: mixed peak {peak:.3f}")
    frame = int(0.08 * sr)
    n = len(mixed) // frame
    if n:
        rms = np.sqrt((mixed[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
        info["silence_frac"] = round(float((rms < 1e-3).mean()), 3)

    return issues, info


def utt_for(tl, sc_dir: Path, ev) -> Path:
    """Locate the rendered wav for an event (render names by speech index)."""
    speech_idx = 0
    for e in tl.events:
        if e is ev:
            return sc_dir / "utterances" / f"{speech_idx:02d}_{ev.ch}.wav"
        if e.text and e.kind != "marker":
            speech_idx += 1
    raise LookupError(f"event not found for {ev.ch}@{ev.actual_start_ms}")


# ---------------------------------------------------------------- judge pass

def ref_for_event(ev) -> tuple[str, bool]:
    """Intended text for one rendered event; truncated events contribute their
    estimated spoken prefix. Returns (text, was_truncated)."""
    if not is_truncated(ev):
        return ev.text, False
    words = ev.text.split()
    kept_ms = ev.actual_end_ms - ev.actual_start_ms
    est_full_ms = max(len(words) * MS_PER_WORD_EST, 1)
    frac = min(1.0, kept_ms / est_full_ms)
    k = max(1, int(round(len(words) * frac)))
    return " ".join(words[:k]), True


def load_asr(model_id: str):
    """Open ASR judge (default whisper-large-v3-turbo, MIT). ADR-0002: the
    filter runs open models only, same rule as the data itself."""
    import torch
    from transformers import pipeline

    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device="cuda",
    )


def judge_scenario(pipe, sc_dir: Path, tl) -> dict:
    """Per-utterance round trip: transcribe each rendered utterance wav and
    compare with its own text (truncation-aware). Judging whole stems instead
    makes whisper hallucinate over long silences (timed_task) and mixes
    backchannel tokens into every channel — the per-utterance wavs isolate
    exactly what we want to measure: each TTS render."""
    import librosa
    import soundfile as sf

    per_utt: list[dict] = []
    speech = [e for e in tl.events if e.text and e.kind != "marker"]
    voice_of = tl.meta.get("voices", {})
    for i, ev in enumerate(speech):
        voice = ev.voice or voice_of.get(ev.ch)
        wav_path = sc_dir / "utterances" / f"{i:02d}_{ev.ch}.wav"
        if not wav_path.exists():
            per_utt.append({"ch": ev.ch, "voice": voice,
                            "missing": str(wav_path.name), "wer": 1.0})
            continue
        ref, trunc = ref_for_event(ev)
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if float(np.max(np.abs(audio))) < 1e-4:
            hyp = ""
        else:
            hyp = pipe(audio, language="en", task="transcribe")["text"]
        w = wer(norm_words(ref), norm_words(hyp))
        n_ref = len(norm_words(ref))
        dur_s = len(audio) / sr
        entry = {"ch": ev.ch, "voice": voice, "wer": round(w, 3),
                 "n_ref": n_ref, "dur_s": round(dur_s, 2),
                 "truncated_ref": trunc, "kind": ev.kind,
                 "gate": "duration" if n_ref <= SHORT_REF_WORDS else "wer",
                 "ref": ref[:120], "hyp": str(hyp)[:120]}
        # short refs ("mm-hm", spelled syllables) don't round-trip through
        # ASR tokenization — gate them on duration (TTS babble runs long)
        if n_ref <= SHORT_REF_WORDS and dur_s > SHORT_MAX_S:
            entry["gate_fail"] = f"babble: {dur_s:.1f}s for {n_ref}-word ref"
        per_utt.append(entry)

    gate_fails = [it["gate_fail"] for it in per_utt if "gate_fail" in it]

    # channel / voice / scenario aggregates weighted by ref word count,
    # over WER-gated utterances only
    def agg(items):
        items = [it for it in items if it.get("gate") == "wer"]
        num = den = 0
        for it in items:
            n = it["n_ref"]
            num += it["wer"] * n
            den += n
        return round(num / den, 3) if den else (1.0 if items else 0.0)

    def group(key):
        keys = sorted({it[key] for it in per_utt if it.get(key)})
        return {k: agg([i for i in per_utt if i.get(key) == k]) for k in keys}

    return {
        "per_utterance": per_utt,
        "gate_fails": gate_fails,
        "wer_by_channel": group("ch"),
        "wer_by_voice": group("voice"),
        "wer": agg(per_utt),
    }


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="render.py output dir")
    ap.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--wer-channel", type=float, default=0.30,
                    help="per-channel WER hard fail threshold")
    ap.add_argument("--no-model", action="store_true",
                    help="rule checks only (CPU, no ASR)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data.synthetic.timeline import Timeline

    audio = Path(args.audio)
    rows = [json.loads(l) for l in open(audio / "manifest.jsonl")]
    print(f"{len(rows)} scenarios in {audio}")

    pipe = None if args.no_model else load_asr(args.asr_model)

    n_pass = 0
    with open(audio / "filter_report.jsonl", "w") as rf, \
            open(audio / "train_manifest.jsonl", "w") as tf:
        for row in rows:
            sc_dir = audio / row["dir"]
            tl = Timeline.load(sc_dir / "timeline.actual.json")

            issues, info = rule_checks(tl, sc_dir)

            judged = None
            if pipe is not None:
                judged = judge_scenario(pipe, sc_dir, tl)
                for ch, w in judged["wer_by_channel"].items():
                    if w > args.wer_channel:
                        issues.append(f"wer_{ch}: {w:.3f}")
                issues += judged["gate_fails"]

            verdict = "pass" if not issues else "fail"
            if verdict == "pass":
                n_pass += 1
                tf.write(json.dumps(row, ensure_ascii=False) + "\n")
            rec = {"scenario_id": row["scenario_id"],
                   "scenario_class": row["scenario_class"],
                   "verdict": verdict, "issues": issues, **info,
                   "asr_model": None if pipe is None else args.asr_model,
                   "wer": None if judged is None else judged["wer"],
                   "wer_by_channel": (None if judged is None
                                      else judged["wer_by_channel"]),
                   "asr_detail": judged}
            rf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [{verdict}] {row['scenario_id']}"
                  + (f" :: {issues}" if issues else ""), flush=True)

    print(f"{n_pass}/{len(rows)} passed -> train_manifest.jsonl")


if __name__ == "__main__":
    main()
