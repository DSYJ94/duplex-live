"""Serialize / parse the interleaved micro-turn stream (Issue #2).

Canonical machine format: JSONL, one line per NON-DEFAULT tick (default =
user silent, emission <sil>, speech inactive, no marks). Every line is one
80 ms step of the causal sequence in docs/09 §1 — the audio-in slot is
implicit in the tick index (AuT frame t attaches at training time), the
emission is the Thinker's x_t target.

Compact view (`to_compact`): one char per tick for eyeballing timelines:
  '.' both silent   'u' user speaking   'A' assistant voicing (no new word)
  'W' word emission 'Y' <yield>         'D' <delegate>      'T' <t+Ns>
  '|' barge-in mark (overlays the tick's other state on its own column)
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer.grid import GRID_MS, GridSequence, Tick  # noqa: E402
from tokenizer.vocab import DELEGATE, SIL, YIELD  # noqa: E402


def to_jsonl(grid: GridSequence) -> str:
    lines = [json.dumps({
        "scenario_id": grid.scenario_id,
        "scenario_class": grid.scenario_class,
        "n_ticks": grid.n_ticks,
        "audio": grid.audio,
        "meta": grid.meta,
    }, ensure_ascii=False)]
    for t in grid.ticks:
        if t.is_default():
            continue
        rec = {"t": t.idx}
        if t.user_in:
            rec["in"] = 1
        if t.emission != SIL:
            rec["out"] = t.emission
        if t.speech_active:
            rec["voi"] = 1
        if t.marks:
            rec["mark"] = t.marks
        lines.append(json.dumps(rec, ensure_ascii=False))
    return "\n".join(lines) + "\n"


def from_jsonl(text: str) -> GridSequence:
    head, *rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    grid = GridSequence(head["scenario_id"], head["scenario_class"],
                        head["n_ticks"],
                        [Tick(i, i * GRID_MS) for i in range(head["n_ticks"])],
                        head.get("audio", {}), head.get("meta", {}))
    for r in rows:
        t = grid.ticks[r["t"]]
        t.user_in = bool(r.get("in", 0))
        t.emission = r.get("out", SIL)
        t.speech_active = bool(r.get("voi", 0))
        t.marks = r.get("mark", [])
    return grid


def to_compact(grid: GridSequence) -> str:
    out = []
    for t in grid.ticks:
        if t.emission == YIELD:
            c = "Y"
        elif t.emission == DELEGATE:
            c = "D"
        elif t.emission.startswith("<t+"):
            c = "T"
        elif t.emission != SIL:
            c = "W"
        elif t.speech_active:
            c = "A"
        elif t.user_in:
            c = "u"
        else:
            c = "."
        out.append("|" if t.marks else c)
    return "".join(out)


def write(grid: GridSequence, path: Path) -> None:
    Path(path).write_text(to_jsonl(grid))


def load(path: Path) -> GridSequence:
    return from_jsonl(Path(path).read_text())
