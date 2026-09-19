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


def measure(test: str, marks: dict, out_wav: np.ndarray, sr: int,
            in_wav: np.ndarray | None = None) -> dict:
    """Compute the metric set for one replayed test. Times are seconds;
    *_ms metrics are the headline numbers."""
    out_mask = activity(frame_rms(out_wav, sr))
    res = {"test": test, "out_speech_s": round(speech_seconds(out_mask), 2)}

    if test == "t2":
        barge_t = marks["barge"][0]
        bt = round(barge_t * 1000 / GRID_MS)
        # barge -> model went silent
        sil = offset_after(out_mask, bt - 2)
        res["barge_to_silence_ms"] = (sil - bt) * GRID_MS if sil else None
        # barge -> model speaks again (the revised answer)
        reply = onset_after(out_mask, bt + 6)
        res["reply_after_barge_ms"] = (reply - bt) * GRID_MS if reply else None
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
        part2_t = round(marks["part"][1] * 1000 / GRID_MS)
        # model answer to part1 should END (turn complete), and the total
        # output should include a reply after part2 — resume driven by user
        reply2 = onset_after(out_mask, part2_t - 2)
        res["answered_part2"] = bool(reply2)
        # premature answer between part1 and part2 (before user finished):
        early = onset_after(out_mask, round(marks["part"][0] * 1000 / GRID_MS) + 8)
        res["premature_answer"] = bool(early and early < part2_t - 10)

    elif test == "t5":
        res["false_engagement_s"] = res["out_speech_s"]

    elif test == "t1":
        lats = []
        for q_s in marks["q"]:
            on = onset_after(out_mask, round(q_s * 1000 / GRID_MS) + 2)
            if on is not None:
                lats.append((on - round(q_s * 1000 / GRID_MS)) * GRID_MS)
        if lats:
            res["response_latency_ms"] = sorted(lats)
            res["response_latency_p50_ms"] = int(np.median(lats))
    return res


def framehash_ledger(in_wav: np.ndarray, sr: int, chunk_s: float = 1.0) -> list[dict]:
    """qwen-live-style per-chunk stimulus fingerprints for traceability."""
    n = int(sr * chunk_s)
    out = []
    for c in range(len(in_wav) // n):
        out.append({"chunk": c, "t_s": c * chunk_s,
                    "sha16": chunk_hash(in_wav[c * n:(c + 1) * n])})
    return out
