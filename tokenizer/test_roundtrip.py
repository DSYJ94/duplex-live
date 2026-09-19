"""Round-trip + invariant tests for the micro-turn grid (Issue #2).

Run:  python -m tokenizer.test_roundtrip [--batch runs/synth-big2/audio]

1. Synthetic timelines across all scenario classes build grids whose stats
   match the source events (word counts, yields, delegates).
2. serialize -> parse reconstructs the grid exactly.
3. Real factory batches (local copies) round-trip; overlap/yield invariants
   hold (yield lands on the barge tick; user/assistant overlap ticks exist
   where the timeline says so).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.synthetic.scenarios import GENERATORS  # noqa: E402
from data.synthetic.timeline import Timeline      # noqa: E402
from tokenizer import serialize                   # noqa: E402
from tokenizer.grid import GRID_MS, build_grid    # noqa: E402
from tokenizer.vocab import BARGE_IN, DELEGATE, SIL, YIELD  # noqa: E402


def _fill_actual(tl: Timeline, rng: random.Random) -> None:
    """Fabricate actual timings the renderer would produce (incl. a cut)."""
    speech = [e for e in tl.events if e.text and e.kind != "marker"]
    for e in speech:
        if e.actual_start_ms is None:
            e.actual_start_ms = e.quantized_start()
        dur = 800 + int(len(e.text) * 80 * rng.uniform(0.85, 1.15))
        e.actual_end_ms = e.actual_start_ms + round(dur / GRID_MS) * GRID_MS
    for e in speech:
        if e.kind == "barge_in":
            for v in speech:
                if v.ch != e.ch and v.actual_start_ms < e.actual_start_ms < v.actual_end_ms:
                    v.actual_end_ms = round(e.actual_start_ms / GRID_MS) * GRID_MS
                    v.note = (v.note + "; " if v.note else "") + \
                        f"truncated at barge-in {e.actual_start_ms}ms"


def test_synthetic_roundtrip() -> None:
    rng = random.Random(7)
    for cls, gen in GENERATORS.items():
        tl = gen(rng, 0)
        _fill_actual(tl, rng)
        grid = build_grid(tl)
        rt = serialize.from_jsonl(serialize.to_jsonl(grid))
        assert serialize.to_jsonl(rt) == serialize.to_jsonl(grid), cls

        # word-count invariant: non-truncated events emit every word;
        # truncated events emit a strict prefix (post-cut words retracted)
        asst = [e for e in tl.events
                if e.ch == "assistant" and e.kind != "marker"]
        full = sum(len(e.text.split()) for e in asst
                   if "truncated at barge-in" not in (e.note or ""))
        trunc = [e for e in asst if "truncated at barge-in" in (e.note or "")]
        got = rt.stats()["words"]
        assert got >= full, (cls, got, full)
        assert got <= full + sum(len(e.text.split()) - 1 for e in trunc), \
            (cls, got, full, len(trunc))

        # yield lands on cut events' end tick
        n_cut = sum(1 for e in tl.events
                    if "truncated at barge-in" in (e.note or ""))
        n_yield = sum(1 for t in rt.ticks if t.emission == YIELD)
        assert n_yield == n_cut, (cls, n_yield, n_cut)

        # barge markers present
        n_barge = sum(1 for e in tl.events if e.kind == "barge_in")
        n_mark = sum(1 for t in rt.ticks if BARGE_IN in t.marks)
        assert n_mark == n_barge, (cls, n_mark, n_barge)

        # delegate markers
        n_del = sum(1 for e in tl.events
                    if e.kind == "marker" and "delegate" in (e.note or ""))
        n_delg = sum(1 for t in rt.ticks if t.emission == DELEGATE)
        assert n_delg == n_del, (cls, n_delg, n_del)
    print("synthetic round-trip + invariants: OK (all classes)")


def test_real_batch(batch: Path) -> None:
    from tokenizer.grid import from_scenario_dir

    n = 0
    for sc in sorted(batch.iterdir()):
        if not (sc / "timeline.actual.json").exists():
            continue
        grid = from_scenario_dir(sc)
        rt = serialize.from_jsonl(serialize.to_jsonl(grid))
        assert serialize.to_jsonl(rt) == serialize.to_jsonl(grid), sc.name

        tl = Timeline.load(sc / "timeline.actual.json")
        # yields == cuts; words = full non-truncated + strict prefixes
        n_cut = sum(1 for e in tl.events
                    if "truncated at barge-in" in (e.note or ""))
        assert sum(1 for t in rt.ticks if t.emission == YIELD) == n_cut, sc.name
        asst = [e for e in tl.events
                if e.ch == "assistant" and e.kind != "marker"]
        full = sum(len(e.text.split()) for e in asst
                   if "truncated at barge-in" not in (e.note or ""))
        trunc = [e for e in asst if "truncated at barge-in" in (e.note or "")]
        got = rt.stats()["words"]
        assert got >= full, (sc.name, got, full)
        assert got <= full + sum(len(e.text.split()) - 1 for e in trunc), sc.name
        # overlap ticks > 0 wherever the timeline has cross-channel overlap
        ov_ms = 0
        sp = [e for e in tl.events if e.text and e.kind != "marker"]
        for i, a in enumerate(sp):
            for b in sp[i + 1:]:
                if a.ch != b.ch:
                    ov_ms += max(0, min(a.actual_end_ms, b.actual_end_ms)
                                 - max(a.actual_start_ms, b.actual_start_ms))
        assert (rt.stats()["overlap_ticks"] > 0) == (ov_ms >= GRID_MS), sc.name
        n += 1
    print(f"real batch round-trip: OK ({n} scenarios from {batch})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None)
    args = ap.parse_args()
    test_synthetic_roundtrip()
    if args.batch:
        test_real_batch(Path(args.batch))


if __name__ == "__main__":
    main()
