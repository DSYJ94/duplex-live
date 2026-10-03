"""TIMING AUDIT — check the answer key's word timings against the real
audio (fail-closed data rule, owner directive 2026-10-02, part 2).

Plain words: training teaches the model WHEN each assistant word happens
(the 80 ms tick grid). The grid places each word inside its clip by a
characters-proportional estimate (v0 alignment) — nobody ever checked
those estimates against the actual sound. This module does: open whisper
(the same model the quality judge uses) transcribes each assistant clip
WITH word timestamps, and every answer-key word is compared with the time
its sound really starts. Offsets land in timing_report.jsonl per
conversation; the quality stamp (data.synthetic.quality) turns them into
the timing=ok / timing=fail field every training job asserts on.

Only ASSISTANT words are checked — user words are not answer-key labels
(user speech is the model's INPUT, and the input is the real audio
itself, so input timing is correct by construction).

Run (Isambard, GH200, pylibs overlay):
  python -m data.synthetic.timing_audit --batch runs/synth-scale1
  python -m data.synthetic.quality --batch runs/synth-scale1 \
      --timing runs/synth-scale1/timing_report.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

GRID_MS = 80
DEFAULT_BAR_MS = 200  # 2.5 ticks; first run measures the true distribution
# and the bar gets re-set from that evidence before training batches stamp.


def load_asr(model_id: str):
    """Same open judge as the filter (ADR-0002), asked for word timestamps."""
    import torch
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=model_id,
                    torch_dtype=torch.bfloat16, device="cuda")


def grid_words_by_event(tl) -> tuple[list, list[tuple[int, int, str]]]:
    """Replay build_grid's assistant-word placement (same helpers, same
    collision rule) while remembering which EVENT each placed word came
    from. Returns (assistant_events, [(event_index, tick_idx, word), ...]).
    KEEP IN SYNC with tokenizer/grid.py build_grid — this replay exists
    only to attribute grid words to clips; build_grid stays the canonical
    answer key."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tokenizer.grid import GRID_MS as G
    from tokenizer.vocab import DELEGATE, SIL, YIELD, timing_token

    speech = [e for e in tl.events if e.text and e.kind != "marker"]
    end_ms = max((e.actual_end_ms or e.quantized_start() + 800)
                 for e in tl.events)
    n = round(end_ms / G) + 1
    emis = [SIL] * n

    asst = sorted((e for e in speech if e.ch == "assistant"),
                  key=lambda e: e.actual_start_ms)
    prev_end = None
    for ev in asst:
        if prev_end is not None:
            tok = timing_token(ev.actual_start_ms - prev_end)
            if tok:
                i = round(ev.actual_start_ms / G)
                if 0 <= i < n and emis[i] == SIL:
                    emis[i] = tok
        prev_end = ev.actual_end_ms
    for ev in asst:
        if "truncated at barge-in" not in (ev.note or ""):
            continue
        i = round(ev.actual_end_ms / G)
        if 0 <= i < n:
            emis[i] = YIELD
    for m in (e for e in tl.events if e.kind == "marker"):
        if "delegate" in (m.note or ""):
            i = round(m.quantized_start() / G)
            if 0 <= i < n:
                emis[i] = DELEGATE

    from tokenizer.grid import _word_ticks

    out: list[tuple[int, int, str]] = []

    def free(i: int) -> bool:
        return emis[i] == SIL or emis[i].startswith("<t+")

    for ei, ev in enumerate(asst):
        trunc = "truncated at barge-in" in (ev.note or "")
        for tick_i, w in _word_ticks(ev.text, ev.actual_start_ms,
                                     ev.actual_end_ms, trunc, ev=ev):
            i = tick_i
            while i < n and not free(i):
                i += 1
            if i < n:
                emis[i] = w
                out.append((ei, i, w))
    return asst, out


def _bare(w: str) -> str:
    """' Word,' -> 'word' (identity check only; timing is the point)."""
    return "".join(c for c in w.lower() if c.isalpha())


def audit_scenario(pipe, sc_dir: Path, bar_ms: float,
                   relabel: bool = False) -> dict:
    """Word-timestamp check for ONE conversation. Returns the
    timing_report.jsonl record (timing: ok|fail). With relabel=True,
    assistant clips whose grid words ALL matched the ASR words get their
    measured onsets written into the timeline (ev.word_onsets_ms) — the
    answer key then follows the real audio instead of the char-
    proportional estimate (fail-closed data rule: re-label from measured
    times) — and the scenario is re-measured from the saved timeline to
    confirm what the grid will actually hold."""
    import soundfile as sf

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data.synthetic.timeline import Timeline
    from tokenizer.grid import GRID_MS as G

    tl = Timeline.load(sc_dir / "timeline.actual.json")
    asst, placed = grid_words_by_event(tl)
    speech = [e for e in tl.events if e.text and e.kind != "marker"]
    idx_of = {id(ev): i for i, ev in enumerate(speech)}

    by_ev: dict[int, list[tuple[int, str]]] = {}
    for ei, tick, w in placed:
        by_ev.setdefault(ei, []).append((tick, w))

    clips: list[dict] = []
    all_offsets: list[float] = []
    relabeled = 0
    for ei, ev in enumerate(asst):
        words = by_ev.get(ei, [])
        if len(words) < 2:
            continue  # <2 placed words: nothing to align
        wav_path = sc_dir / "utterances" / f"{idx_of[id(ev)]:02d}_{ev.ch}.wav"
        if not wav_path.exists():
            clips.append({"idx": idx_of[id(ev)], "ch": ev.ch,
                          "note": "missing wav", "n": 0})
            continue
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            # TTS wavs are 24 kHz; whisper's extractor assumes 16 kHz —
            # without resampling every word time is stretched 1.5x
            # (smoke lesson, 2026-10-02: p50 offsets of whole seconds)
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if len(audio) / 16000 > 30:
            continue
        out = pipe(audio, language="en", task="transcribe",
                   return_timestamps="word")
        hyp = [(c["text"], c["timestamp"][0]) for c in out.get("chunks", [])
               if c.get("timestamp") and c["timestamp"][0] is not None]
        n = min(len(words), len(hyp))
        offsets: list[float] = []
        matched: list[tuple[int, float]] = []  # (grid word idx, onset s)
        for k in range(n):
            tick, w = words[k]
            txt, t0 = hyp[k]
            if _bare(w) != _bare(txt):
                continue  # index misalignment: skip the pair
            offsets.append(abs(tick * G - (ev.actual_start_ms + t0 * 1000)))
            matched.append((k, t0))
        entry = {"idx": idx_of[id(ev)], "ch": ev.ch, "n_grid": len(words),
                 "n_asr": len(hyp), "n_checked": len(offsets),
                 "truncated": "truncated at barge-in" in (ev.note or "")}
        if offsets:
            entry["p50_ms"] = round(statistics.median(offsets), 1)
            entry["max_ms"] = round(max(offsets), 1)
            all_offsets.extend(offsets)
        if relabel and len(matched) == len(words):
            # every answer-key word has a measured onset: re-label.
            # Truncated events keep only the voiced prefix (cut audio).
            ev.word_onsets_ms = [round(t0 * 1000) for _, t0 in matched]
            entry["relabeled"] = True
            relabeled += 1
        clips.append(entry)

    if relabel and relabeled:
        tl.save(sc_dir / "timeline.actual.json")
        base = audit_scenario(pipe, sc_dir, bar_ms, relabel=False)
        base["relabeled"] = True
        base["n_relabeled"] = relabeled
        return base

    offs = sorted(all_offsets)

    def pct(p: float) -> float:
        return round(offs[min(len(offs) - 1, int(len(offs) * p))], 1) if offs else -1

    worst = max((c for c in clips if c.get("max_ms") is not None),
                key=lambda c: c["max_ms"], default=None)
    ok = bool(offs) and pct(0.95) <= bar_ms and offs[-1] <= 2 * bar_ms
    return {"scenario_id": tl.scenario_id,
            "scenario_class": tl.scenario_class,
            "asr_model": "whisper-word-timestamps",
            "offset_bar_ms": bar_ms,
            "n_words_checked": len(offs),
            "p50_ms": pct(0.50), "p95_ms": pct(0.95), "max_ms": offs[-1] if offs else -1,
            "worst_clip": worst, "clips": clips,
            "timing": "ok" if ok else "fail"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True,
                    help="factory batch dir (train_manifest.jsonl + "
                         "<scenario>/timeline.actual.json)")
    ap.add_argument("--asr-model",
                    default="openai/whisper-large-v3-turbo")
    ap.add_argument("--offset-bar-ms", type=float, default=DEFAULT_BAR_MS,
                    help="answer-key word may miss the real word start by "
                         "at most this many ms (p95 of a conversation)")
    ap.add_argument("--limit", type=int, default=None,
                    help="audit only the first N conversations (smoke)")
    ap.add_argument("--relabel", action="store_true",
                    help="write measured word onsets into the timeline "
                         "(re-label the answer key from the real audio), "
                         "then re-measure to confirm — fail-closed rule: "
                         "re-label from measured times, reject if that "
                         "fails")
    args = ap.parse_args()

    from data.synthetic.quality import SCHEMA  # schema echo only

    batch = Path(args.batch)
    rows = [json.loads(l) for l in open(batch / "train_manifest.jsonl")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[timing] auditing {len(rows)} conversations "
          f"(bar: p95 offset <= {args.offset_bar_ms:g} ms)", flush=True)
    pipe = load_asr(args.asr_model)
    n_ok = n_fail = 0
    hist: dict[str, int] = {}
    with open(batch / "timing_report.jsonl", "w") as tf:
        for row in rows:
            sc_dir = batch / row["dir"]
            rec = audit_scenario(pipe, sc_dir, args.offset_bar_ms,
                                 relabel=args.relabel)
            rec["schema"] = SCHEMA
            tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tf.flush()
            if rec["timing"] == "ok":
                n_ok += 1
            else:
                n_fail += 1
            for c in rec["clips"]:
                m = c.get("max_ms")
                if m is None:
                    continue
                b = ("<80" if m < 80 else "80-160" if m < 160
                     else "160-240" if m < 240 else "240-400" if m < 400
                     else ">400")
                hist[b] = hist.get(b, 0) + 1
            note = (f", re-labeled {rec['n_relabeled']} clips"
                    if rec.get("relabeled") else "")
            print(f"[timing] {rec['scenario_id']}: {rec['timing']} "
                  f"(checked {rec['n_words_checked']} words, p95 "
                  f"{rec['p95_ms']}ms, max {rec['max_ms']}ms{note})",
                  flush=True)
    print(f"[timing] done: {n_ok} ok, {n_fail} FAIL "
          f"(bar p95 <= {args.offset_bar_ms:g}ms)"
          + (" — answer keys re-labeled from measured onsets"
             if args.relabel else ""))
    print("[timing] worst-offset distribution over clips "
          "(evidence for setting the bar):")
    for b in ("<80", "80-160", "160-240", "240-400", ">400"):
        print(f"[timing]   {b}ms: {hist.get(b, 0)} clips")
    print("[timing] next: re-stamp quality with --timing timing_report.jsonl")


if __name__ == "__main__":
    main()
