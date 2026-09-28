"""CPU regression test for layera.freerun_diag.word_onset_stats.

Guards the two metric bugs the capacity/self-history jobs paid for:
  1. the vacuous +-1s metric (a fully MUTED run scored 14/14 "near"
     because each gold tick sat inside its own window) — job 6814321;
  2. the NameError left by a metric refactor (job 6816545's evals
     died at the WORD ONSETS print, after training had completed).

Run: python -m evals.harness.test_freerun_metrics
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layera.freerun_diag import (  # noqa: E402
    speech_pr, speech_run_onsets, user_block_mask, word_onset_stats)
from tokenizer.vocab import (  # noqa: E402
    ALL_SPECIAL, SIL, SPEAK, is_speech_emission)


def check(name, got, want):
    assert got == want, f"{name}: got {got}, want {want}"
    print(f"  ok {name}: {got}")


def main() -> None:
    gold = [50, 60, 70, 80, 100, 110, 120, 130, 140, 150, 160, 170,
            180, 190]  # 14 gold word ticks

    # muted run: NO speech at all — exact 0, near 0 (was vacuously 14)
    check("muted", word_onset_stats(gold, []), (0, 0, 0, 14))

    # late answers + tail chatter: speech 6 ticks (~0.5 s) after every
    # gold tick — exact 0, near 14; two far ticks count as chatter
    spoken = [g + 6 for g in gold] + [250, 260]
    exact, near, false_speech, n_word = word_onset_stats(gold, spoken)
    check("late answers exact", exact, 0)
    check("late answers near", near, 14)
    check("late answers chatter", false_speech, 2)
    check("late answers n_word", n_word, 14)

    # perfect run: speaks exactly at gold ticks — exact 14, near 14,
    # zero chatter
    check("perfect", word_onset_stats(gold, gold), (14, 14, 0, 14))

    # one late answer: speech 8 ticks (640 ms) after gold tick 50 —
    # not exact, but inside its +-1s window (ticks 48..62)
    check("late-but-close", word_onset_stats([50], [58]), (0, 1, 0, 1))

    # window asymmetry pinned (2026-09-28 audit): "near" = ticks
    # w-2..w+12 (160 ms back, 1040 ms forward at 80 ms/tick) — early
    # is judged harder than late, by design (early = interruption)
    check("near at w-2", word_onset_stats([50], [48]), (0, 1, 0, 1))
    check("chatter at w-3", word_onset_stats([50], [47]), (0, 0, 1, 1))
    check("near at w+12", word_onset_stats([50], [62]), (0, 1, 0, 1))
    check("chatter at w+13", word_onset_stats([50], [63]), (0, 0, 1, 1))
    # window clamps at tick 0 (no negative indices)
    check("clamp at 0", word_onset_stats([1], [0]), (0, 1, 0, 1))

    # ---- speech_pr: the P/R/F1 arithmetic (legacy AND v3 lines) ----
    check("pr perfect", speech_pr({1, 2}, {1, 2}), (1.0, 1.0, 1.0))
    check("pr partial", speech_pr({1, 2, 3}, {1, 2}), (2 / 3, 1.0, 0.8))
    check("pr empty spoken", speech_pr(set(), {1}), (0.0, 0.0, 0.0))
    check("pr empty gold", speech_pr({1}, set()), (0.0, 0.0, 0.0))

    # ---- the ONE speech-label definition (tokenizer.vocab) ----
    # gate labels (train_gate), dagger rollout labels (train_gate) and
    # run-onset/v3 metrics (freerun_diag) must answer identically
    assert is_speech_emission("hello") and is_speech_emission(SPEAK)
    assert not is_speech_emission(SIL)
    assert not any(is_speech_emission(s)
                   for s in ALL_SPECIAL if s != SPEAK)
    print("  ok is_speech_emission: word/<speak> speech; "
          "sil/timing/yield/other specials quiet")

    # ---- stopgap user-gate mask (user_block_mask) ----
    # 1280 samples/tick; a loud burst at ticks 10-14 blocks itself plus
    # hold=3 ticks after, nothing before
    pcm = np.zeros(30 * 1280, dtype=np.float32)
    pcm[10 * 1280:15 * 1280] = 0.4
    b = user_block_mask(pcm, 30, 3)
    assert not b[9] and b[10] and b[14] and b[17] and not b[18], b
    # constant floor: p25 == p95 -> nothing hot -> nothing blocked
    b2 = user_block_mask(np.full(20 * 1280, 0.01, dtype=np.float32),
                         20, 5)
    assert not b2.any(), b2
    print("  ok user_block_mask: burst blocks hold=3 after, "
          "constant floor blocks nothing")

    # ---- v2 contract: speech_run_onsets --------------------------------
    # a run = consecutive word/<speak> ticks; every other special is
    # quiet. runs start where quiet -> speech transitions happen.
    em = [SIL, SIL, "hello", SPEAK, SPEAK, SIL, "<t+5s>",
          "yes", SPEAK, "<yield>", SIL]
    check("v2 run onsets", speech_run_onsets(em), [2, 7])
    # v1-style grid (no <speak> yet): words alone are 1-tick runs
    check("v1 words only", speech_run_onsets([SIL, "a", SIL, "b"]), [1, 3])
    # fully muted gold: no onsets
    check("all quiet", speech_run_onsets([SIL, "<t+10s>", SIL]), [])
    # speech at tick 0 starts a run; yield ends it
    check("leading run",
          speech_run_onsets(["hi", SPEAK, "<yield>", SIL, "ok"]), [0, 4])
    # <speak> really is in the registered specials (serving + training
    # must agree on this) and sil is not speech for run purposes
    assert SPEAK in ALL_SPECIAL and SIL in ALL_SPECIAL
    print("  ok <speak> registered in ALL_SPECIAL")

    print("test_freerun_metrics: all green")


if __name__ == "__main__":
    main()
