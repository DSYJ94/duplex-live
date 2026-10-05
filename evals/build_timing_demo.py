"""TIMING-DEMO picks builder (2026-10-05, the honest audible demo).

Plain words: the trunk model now decides WHEN to speak (the ear-clock
fix made its turn-starts land on time — 7/7 within 1 s on held-out
lessons), but its free-run word CONTENT is degenerate (digit-soup:
93% of its picks are single characters like "0" and "9" — the TTS read
them as the groaning drone the owner heard 2026-10-05). Content was
never trained in free-run: phase-1 is teacher-forced, words were
always handed over. This builder makes the demo honest about exactly
that split:

  the MODEL'S timing + the SCRIPT'S words.

For each gold assistant utterance, if the model started a word-run
within +-1.5 s of the utterance's start, the utterance's words are
re-emitted starting at the MODEL'S first pick tick (its actual onset,
early or late). Turns the model never started are DROPPED (missed
turns, audible as gaps); model starts with no gold turn are skipped
(over-speaking, reported in the numbers instead of faked audio).

Output: a picks dir in the same format freerun_diag --save-picks
writes, ready for evals.render_model_audio.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WINDOW_TICKS = 19  # +-1.5 s around the gold start, in 80 ms ticks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True,
                    help="dir of freerun pick jsons (digit soup lives here)")
    ap.add_argument("--batch", required=True,
                    help="factory batch dir (timelines + shard-free)")
    ap.add_argument("--out", required=True, help="dir for the cleaned picks")
    args = ap.parse_args()

    from data.synthetic.timeline import Timeline
    from tokenizer.grid import GRID_MS, _word_ticks

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for pf in sorted(Path(args.picks).glob("*.json")):
        d = json.loads(pf.read_text())
        sid = d["scenario_id"]
        picks = d["words"]
        sc_dir = Path(args.batch) / sid
        tl = Timeline.load(sc_dir / "timeline.actual.json")
        asst = [e for e in tl.events
                if e.ch == "assistant" and e.text and e.kind != "marker"]
        n = d["n_ticks"]

        pick_ticks = sorted(w["tick"] for w in picks)
        used = set()
        new_words, started, missed = [], 0, 0
        for ev in asst:
            t0 = round(ev.actual_start_ms / GRID_MS)
            lo, hi = t0 - WINDOW_TICKS, t0 + WINDOW_TICKS
            hit = next((t for t in pick_ticks
                        if lo <= t <= hi and t not in used), None)
            if hit is None:
                missed += 1
                continue
            used.add(hit)
            started += 1
            trunc = "truncated at barge-in" in (ev.note or "")
            # re-place the utterance's words from the MODEL'S onset tick
            span_ms = ev.actual_end_ms - ev.actual_start_ms
            placed = _word_ticks(ev.text, hit * GRID_MS,
                                 hit * GRID_MS + span_ms, trunc, ev=ev)
            for tick, w in placed:
                if tick < n:
                    new_words.append({"tick": tick, "ids": [], "text": w})
        json.dump({"scenario_id": sid, "n_ticks": n, "words": new_words,
                   "timing_demo": True,
                   "turns_started": started, "turns_missed": missed},
                  open(out / pf.name, "w"), indent=1)
        print(f"[timing-demo] {sid}: {started}/{started + missed} gold turns "
              f"started by the model, re-spoken at ITS onsets "
              f"({len(new_words)} words)", flush=True)
    print("[timing-demo] done — render with evals.render_model_audio")


if __name__ == "__main__":
    main()
