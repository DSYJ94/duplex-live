"""Regression tests for the repaired gate scoring (2026-09-22, Codex F1).

CPU-only (numpy + soundfile, no model, no GPU). Builds synthetic
recordings in the replay layout and asserts the scoring does what the
repair promised:

  1. three-tone cheat  — tone at each question START — must FAIL t1
     (it used to PASS 5/5);
  2. ideal t1 trace    — answer just after each question END — must PASS;
  3. t2 reply anchored at barge END — a 300 ms-after-barge-end reply
     PASSES (the start-anchor used to false-FAIL it at ~5 s);
  4. t4 talking over the user's part 2 must FAIL; waiting must PASS.

  python -m evals.harness.test_gate_regressions
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from evals.harness.protocol import T_PROTOCOL
from evals.harness.run import replay

SR_IN, SR_OUT = 16000, 24000
TONE_HZ = 220
AMP = 0.06  # well above the 1e-3 activity floor


def _burst(dur_s: float, sr: int) -> np.ndarray:
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    w = AMP * np.sin(2 * np.pi * TONE_HZ * t)
    edge = min(int(0.004 * sr), n // 4)
    env = np.ones(n, np.float32)
    if edge:
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
        env[:edge] = ramp
        env[-edge:] = ramp[::-1]
    return (w * env).astype(np.float32)


def _write_case(root: Path, test: str, in_events, out_events) -> None:
    """in_events/out_events: [(start_s, dur_s)] — user stimulus and model
    speech on the t-protocol timeline, written as replay-layout wavs."""
    d = root / test
    d.mkdir(parents=True, exist_ok=True)
    total_s = T_PROTOCOL[test]["total_s"]
    tin = np.zeros(int(total_s * SR_IN), np.float32)
    tout = np.zeros(int(total_s * SR_OUT), np.float32)
    for s, dur in in_events:
        b = _burst(dur, SR_IN)
        i0 = int(s * SR_IN)
        tin[i0:i0 + len(b)] += b
    for s, dur in out_events:
        b = _burst(dur, SR_OUT)
        i0 = int(s * SR_OUT)
        tout[i0:i0 + len(b)] += b
    sf.write(d / "user_in.wav", np.clip(tin, -1, 1), SR_IN)
    sf.write(d / "model_out.wav", np.clip(tout, -1, 1), SR_OUT)


def _score(root: Path, test: str) -> dict:
    rows = replay(root, [test])
    return rows[0]


def main() -> int:
    fails = []
    q_starts = T_PROTOCOL["t1"]["marks"]["q"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # 1. the cheat: tone exactly at each question start, nothing else
        _write_case(root, "t1", [(q, 2.5) for q in q_starts],
                    [(q, 0.24) for q in q_starts])
        m = _score(root, "t1")
        if "FAIL" not in m["verdict"]:
            fails.append(f"cheat passed: {m['verdict']}")
        if m["n_answered"] != 0:
            fails.append(f"cheat answered {m['n_answered']} (want 0)")
        if m["premature_over_questions"] != 10:
            fails.append(f"cheat premature={m['premature_over_questions']} (want 10)")

        # 2. ideal: answer 400 ms after each question END, 1 s long
        _write_case(root, "t1", [(q, 2.5) for q in q_starts],
                    [(q + 2.9, 1.0) for q in q_starts])
        m = _score(root, "t1")
        if not m["verdict"].startswith("PASS"):
            fails.append(f"ideal t1 failed: {m['verdict']} "
                         f"(answered {m['n_answered']}, p50={m.get('response_latency_p50_ms')})")
        if m["n_answered"] != 10:
            fails.append(f"ideal t1 answered {m['n_answered']}/10")

        # 3. t2: barge ends at 20.8 s; reply 300 ms later must PASS
        _write_case(root, "t2", [(2, 13), (16, 4.8)],
                    [(15.4, 0.84), (21.1, 1.0)])
        m = _score(root, "t2")
        if not m["verdict"].startswith("PASS"):
            fails.append(f"ideal t2 failed: {m['verdict']} "
                         f"(reply={m.get('reply_after_barge_ms')}, "
                         f"sil={m.get('barge_to_silence_ms')})")
        r = m.get("reply_after_barge_ms")
        if r is None or r > 800:
            fails.append(f"t2 reply after END = {r} ms (want <= 800)")

        # 4a. t4 talk-over: answers part1 then keeps going through part2
        _write_case(root, "t4", [(2, 3), (9, 3)], [(5.5, 7.0)])
        m = _score(root, "t4")
        if "FAIL" not in m["verdict"]:
            fails.append(f"t4 talk-over passed: {m['verdict']}")
        if not m["premature_answer"] or m["answered_part2"]:
            fails.append(f"t4 talk-over flags wrong: premature={m['premature_answer']} "
                         f"answered={m['answered_part2']}")

        # 4b. t4 ideal: wait for part2 to finish, then answer
        _write_case(root, "t4", [(2, 3), (9, 3)], [(12.4, 1.0)])
        m = _score(root, "t4")
        if not m["verdict"].startswith("PASS"):
            fails.append(f"t4 ideal failed: {m['verdict']}")

    if fails:
        print("GATE REGRESSION: FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("GATE REGRESSION: all cases behave as repaired (cheat fails, "
          "honest traces pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
