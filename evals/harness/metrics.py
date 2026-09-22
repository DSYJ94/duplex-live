"""Energy-based interactivity metrics over (stimulus, output) audio.

Everything is measured on the 80 ms grid — the same cadence as the factory
and the tokenizer, so eval numbers and training-data gates are directly
comparable (the yield-latency metric here mirrors the factory's
yield_latency gate).
"""

from __future__ import annotations

import hashlib

import numpy as np

GRID_MS = 80


def frame_rms(x: np.ndarray, sr: int) -> np.ndarray:
    n = int(sr * GRID_MS / 1000)
    m = len(x) // n
    if m == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt((x[: m * n].reshape(m, n) ** 2).mean(axis=1))


def activity(rms: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    return rms > floor


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as [start_tick, end_tick)."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def chunk_hash(x: np.ndarray) -> str:
    """frameHash-style diagnostics: first 16 sha256 hex of the PCM bytes."""
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()
                          ).hexdigest()[:16]


def onset_after(mask: np.ndarray, tick: int, min_gap: int = 3) -> int | None:
    """First activity tick >= tick that follows >=240 ms of inactivity."""
    for s, e in runs(mask):
        if s >= tick and (s == 0 or s - _last_end(mask, s) >= min_gap
                          or not mask[:s].any()):
            return s
    return None


def _last_end(mask: np.ndarray, tick: int) -> int:
    prev = [e for s, e in runs(mask[:tick]) if e <= tick]
    return prev[-1] if prev else 0


def offset_after(mask: np.ndarray, tick: int, min_sil: int = 3) -> int | None:
    """First silence-run START (i.e. speech stopped) at/after tick, where
    the silence lasts >= min_sil ticks (filters micro-gaps)."""
    for s, e in runs(~mask):
        if s >= tick and (e - s) >= min_sil:
            return s
    return None


def speech_seconds(mask: np.ndarray) -> float:
    return mask.sum() * GRID_MS / 1000


def stimulus_bounds_from_activity(in_wav: np.ndarray, sr: int,
                                  starts: list[float]) -> list[tuple[float, float]]:
    """(start, end) of each stimulus utterance, read off the input
    activity: the first activity run extending past each mark's onset.
    Run starts are the ground truth for question boundaries too —
    rounding the nominal marks disagrees with the detected run by a tick
    on half-second marks, which let the next question's audio count as
    the previous question's answer. Marks with no audible stimulus fall
    back to (mark, mark)."""
    in_mask = activity(frame_rms(in_wav, sr))
    rr = runs(in_mask)
    out = []
    for s_s in starts:
        st = round(s_s * 1000 / GRID_MS)
        hit = None
        for rs, re_ in rr:
            if re_ > st:  # first run that extends past the mark
                hit = (rs, re_)
                break
        if hit is None:
            hit = (st, st)
        out.append((hit[0] * GRID_MS / 1000, hit[1] * GRID_MS / 1000))
    return out


def measure(test: str, marks: dict, out_wav: np.ndarray, sr: int,
            in_wav: np.ndarray | None = None,
            stim_ends: dict[str, list[float]] | None = None,
            in_sr: int | None = None) -> dict:
    """Compute the metric set for one replayed test. Times are seconds;
    *_ms metrics are the headline numbers.

    Answering latencies are anchored at the END of the user utterance
    (questions/parts/barge) — repaired 2026-09-22: they were anchored at
    utterance START, which demands answering before the user finishes
    and let three well-placed tones pass every test. Ends come from
    stim_ends when given, else derived from the input activity, else
    (no input at all) fall back to the mark itself. in_sr is the input
    sample rate (user_in.wav is 16 kHz, model_out.wav 24 kHz — framing
    the input at the output rate stretched time by 1.5x)."""
    out_mask = activity(frame_rms(out_wav, sr))

    def bounds(kind: str) -> list[tuple[int, int]]:
        """[(start_tick, end_tick)] per stimulus utterance."""
        if stim_ends and kind in stim_ends:
            return [(round(s * 1000 / GRID_MS), round(e * 1000 / GRID_MS))
                    for s, e in stim_ends[kind]]
        if in_wav is not None:
            return [(round(s * 1000 / GRID_MS), round(e * 1000 / GRID_MS))
                    for s, e in stimulus_bounds_from_activity(
                        in_wav, in_sr or sr, marks[kind])]
        return [(round(s * 1000 / GRID_MS),) * 2 for s in marks[kind]]

    res = {"test": test, "out_speech_s": round(speech_seconds(out_mask), 2)}

    if test == "t2":
        barge_t = marks["barge"][0]
        bt = round(barge_t * 1000 / GRID_MS)
        barge_end = bounds("barge")[0][1]
        # barge onset -> model went silent (stop-fast; unchanged anchor)
        sil = offset_after(out_mask, bt - 2)
        res["barge_to_silence_ms"] = (sil - bt) * GRID_MS if sil else None
        # barge END -> model speaks again (the revised answer)
        reply = onset_after(out_mask, barge_end)
        res["reply_after_barge_ms"] = ((reply - barge_end) * GRID_MS
                                       if reply else None)
        res["reply_anchor"] = "barge_end"
        # did the model ever stop BEFORE the barge (it shouldn't — bait)?
        early_sil = offset_after(out_mask, int(marks["bait"][0] * 1000 / GRID_MS) + 12,
                                 min_sil=12)
        res["interrupted_answer"] = bool(early_sil and early_sil < bt - 2)

    elif test == "t3":
        yields = 0
        for bc_s in marks["backchannel"]:
            bt = round(bc_s * 1000 / GRID_MS)
            sil = offset_after(out_mask, bt - 2, min_sil=4)
            if sil and sil < bt + 25:  # dropped within 2s of the backchannel
                yields += 1
        res["backchannel_yields"] = yields

    elif test == "t4":
        part2_end = bounds("part")[1][1]
        # a part-2 answer must START after the user finished part 2
        reply2 = onset_after(out_mask, part2_end)
        res["answered_part2"] = bool(reply2)
        # premature: any onset after part-1 gets going but before part-2
        # ENDS (covers both talking over the stall and over part 2 itself)
        p1_start = bounds("part")[0][0]
        early = onset_after(out_mask, p1_start)
        res["premature_answer"] = bool(early is not None and early < part2_end)
        res["answer_anchor"] = "part2_end"

    elif test == "t5":
        res["false_engagement_s"] = res["out_speech_s"]

    elif test == "t1":
        qb = bounds("q")
        lats, unanswered, premature = [], [], 0
        for i, (qs, qe) in enumerate(qb):
            nxt = qb[i + 1][0] if i + 1 < len(qb) else len(out_mask)
            on = onset_after(out_mask, qe)
            if on is not None and on < nxt:
                lats.append((on - qe) * GRID_MS)
            else:
                unanswered.append(i)
            po = onset_after(out_mask, qs)
            if po is not None and po < qe:
                premature += 1
        res["n_questions"] = len(qb)
        res["n_answered"] = len(lats)
        res["unanswered_questions"] = unanswered
        res["premature_over_questions"] = premature
        if lats:
            res["response_latency_ms"] = sorted(lats)
            res["response_latency_p50_ms"] = int(np.median(lats))
        res["latency_anchor"] = "question_end"
    return res


def framehash_ledger(in_wav: np.ndarray, sr: int, chunk_s: float = 1.0) -> list[dict]:
    """qwen-live-style per-chunk stimulus fingerprints for traceability."""
    n = int(sr * chunk_s)
    out = []
    for c in range(len(in_wav) // n):
        out.append({"chunk": c, "t_s": c * chunk_s,
                    "sha16": chunk_hash(in_wav[c * n:(c + 1) * n])})
    return out
