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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layera.freerun_diag import word_onset_stats


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

    print("test_freerun_metrics: all green")


if __name__ == "__main__":
    main()
