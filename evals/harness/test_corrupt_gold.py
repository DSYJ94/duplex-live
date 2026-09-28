"""CPU regression test for layera.train_p1.corrupt_gold.

Probe 11 (corrupt-gold SH 0.5), gate 1 and gate 2 all stand on this
corruption: training histories get small graded decode-shaped mistakes
while LABELS stay gold. The first test for it lived only in a session
and died with that session (2026-09-28 test audit); this one is
committed. Pins:

  1. only <sil> emission slots are ever overwritten, and only with
     word piece ids that exist in the lesson (decode chatters with
     real pieces, never specials);
  2. labels / audio_pos / length are preserved and the INPUT seq is
     never mutated;
  3. a chatter burst touches <= 12 slots of ONE quiet run;
  4. an early onset writes the FOLLOWING word's FIRST piece into the
     preceding quiet slot (never a random piece);
  5. no pieces or no quiet runs -> returned unchanged;
  6. CORRUPT_STATS counts corrupted samples and overwritten slots.

Run: python -m evals.harness.test_corrupt_gold
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layera.train_p1 import CORRUPT_STATS, corrupt_gold  # noqa: E402

PAD = 0
SIL = 50
SPEAK = 51
PIECES = (20, 21, 30)
IGNORE = -100


def sh_kind(tok: int) -> str:
    if tok in PIECES:
        return "word"
    if tok == SIL:
        return "sil"
    return "speak" if tok == SPEAK else "special"


def seq_six_ticks() -> dict:
    """[pad][sil] [pad][w20 w21] [pad][sil] [pad][w30] [pad][sil] [pad][sil]
    — two word ticks, each preceded by a quiet tick (early-onset
    candidates), plus a trailing two-tick quiet run."""
    ids = [PAD, SIL,
           PAD, 20, 21,
           PAD, SIL,
           PAD, 30,
           PAD, SIL,
           PAD, SIL]
    labels = [IGNORE, SIL,
              IGNORE, 20, 21,
              IGNORE, SIL,
              IGNORE, 30,
              IGNORE, SIL,
              IGNORE, SIL]
    return {"ids": ids, "labels": labels,
            "audio_pos": [0, 2, 5, 7, 9, 11], "scenario_id": "fake6"}


def seq_long_quiet() -> dict:
    """[pad][w20 w21] followed by 20 [pad][sil] ticks — one quiet run
    long enough for the burst bound (<= 12 slots) to bite. The single
    word tick is tick 0, so no early onset is possible (i > 0 guard):
    every difference is burst."""
    ids, labels, pos = [PAD, 20, 21], [IGNORE, 20, 21], [0]
    for _ in range(20):
        pos.append(len(ids))
        ids += [PAD, SIL]
        labels += [IGNORE, SIL]
    return {"ids": ids, "labels": labels, "audio_pos": pos,
            "scenario_id": "fakequiet"}


class ScriptRng:
    """Deterministic rng replaying scripted call answers, so the
    early-onset branch can be tested in isolation (per-seed testing
    can't separate it from a burst hitting the same slot: both write
    legal pieces)."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.i = 0

    def _next(self):
        v = self.answers[self.i]
        self.i += 1
        return v

    def random(self):
        return self._next()

    def choice(self, xs):
        return xs[self._next()]

    def randint(self, a, b):
        return a + self._next() % (b - a + 1)


def main() -> None:
    sil_slots = {1, 6, 10, 12}

    stats0 = list(CORRUPT_STATS)
    n_changed = 0
    total_slots = 0
    for seed in range(120):
        src = seq_six_ticks()
        out = corrupt_gold(src, sh_kind, rng=random.Random(seed))
        # input seq never mutated
        assert src["ids"] == seq_six_ticks()["ids"], seed
        # contract fields and length preserved
        assert out["labels"] == src["labels"], seed
        assert out["audio_pos"] == src["audio_pos"], seed
        assert out["scenario_id"] == src["scenario_id"], seed
        assert len(out["ids"]) == len(src["ids"]), seed

        diffs = {i: b for i, (a, b)
                 in enumerate(zip(src["ids"], out["ids"])) if a != b}
        if diffs:
            n_changed += 1
            total_slots += len(diffs)
            for i, b in diffs.items():
                assert i in sil_slots, (seed, i, b)
                assert b in PIECES, (seed, i, b)
    assert 0 < n_changed < 120, (n_changed, "corruption never/always fires")
    # stats counters (pinned here, with ONLY the six-tick loop run):
    # corrupted samples counted exactly, slots at least
    assert CORRUPT_STATS[0] - stats0[0] == n_changed, \
        (CORRUPT_STATS, stats0, n_changed)
    assert CORRUPT_STATS[1] - stats0[1] >= total_slots
    print(f"  ok CORRUPT_STATS: +{CORRUPT_STATS[0] - stats0[0]} samples, "
          f"+{CORRUPT_STATS[1] - stats0[1]} slots")
    # deterministic: same seed, same corruption
    a = corrupt_gold(seq_six_ticks(), sh_kind, rng=random.Random(7))
    b = corrupt_gold(seq_six_ticks(), sh_kind, rng=random.Random(7))
    assert a["ids"] == b["ids"]
    print(f"  ok six-tick seq: {n_changed}/120 seeds corrupted, "
          f"{total_slots} slots, only sil slots, only real pieces")

    # early onset in isolation: burst aimed at the LAST quiet run
    # (ticks 4-5 -> slot 10), early-onset branch then bleeds the FIRST
    # piece of word tick 1 (slot 3, piece 20) into the preceding quiet
    # slot 1. Script order = corrupt_gold's rng call order:
    # burst-gate, run choice, k, start, piece, early-gate, per-elig gate
    rng = ScriptRng([0.1, 2, 0, 0, 1, 0.1, 0.1])
    out = corrupt_gold(seq_six_ticks(), sh_kind, rng=rng)
    assert out["ids"][10] == 21, out["ids"]   # burst piece on quiet tick 4
    assert out["ids"][1] == 20, out["ids"]    # early onset: FIRST piece
    assert out["ids"][3:5] == [20, 21] and out["ids"][8] == 30
    assert out["labels"] == seq_six_ticks()["labels"]
    print("  ok early onset (scripted): preceding quiet slot <- "
          "following word's first piece; labels untouched")

    # burst bound: one 20-tick quiet run, no early onset possible
    n_changed = 0
    for seed in range(200):
        src = seq_long_quiet()
        out = corrupt_gold(src, sh_kind, rng=random.Random(seed))
        diffs = [i for i, (x, y)
                 in enumerate(zip(src["ids"], out["ids"])) if x != y]
        if diffs:
            n_changed += 1
            assert len(diffs) <= 12, (seed, len(diffs))
            run_slots = {4 + 2 * j for j in range(20)}
            for i in diffs:
                assert i in run_slots, (seed, i)
    assert 0 < n_changed < 200
    print(f"  ok long-quiet seq: {n_changed}/200 seeds corrupted, "
          f"all bursts <= 12 slots inside one run")

    # degenerate inputs: returned unchanged
    all_quiet = {"ids": [PAD, SIL, PAD, SIL], "labels": [IGNORE, SIL, IGNORE, SIL],
                 "audio_pos": [0, 2], "scenario_id": "fq"}
    out = corrupt_gold(all_quiet, sh_kind, rng=random.Random(1))
    assert out["ids"] == all_quiet["ids"]
    only_words = {"ids": [PAD, 20, PAD, 30], "labels": [IGNORE, 20, IGNORE, 30],
                  "audio_pos": [0, 2], "scenario_id": "fw"}
    out = corrupt_gold(only_words, sh_kind, rng=random.Random(1))
    assert out["ids"] == only_words["ids"]
    print("  ok degenerate seqs (no pieces / no quiet runs): unchanged")

    print("test_corrupt_gold: all green")


if __name__ == "__main__":
    main()
