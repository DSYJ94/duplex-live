"""Grid-aligned duplex timeline primitives.

Everything in the synthetic factory lives on the 80 ms micro-turn grid
(ADR-0002 / docs/09). A Timeline is an ordered list of Events on named
channels ("user", "assistant", "background"); the renderer turns it into
overlapping audio and writes actualized start/end times back into the manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

GRID_MS = 80


@dataclass
class Event:
    ch: str               # user | assistant | background
    start_ms: int         # intended start (quantized to GRID_MS on save)
    text: str = ""
    kind: str = "utterance"   # utterance | backchannel | barge_in | pause | marker
    voice: str | None = None  # voice key into render.VOICES; default per channel
    gain: float = 1.0
    note: str = ""
    # filled in by the renderer:
    actual_start_ms: int | None = None
    actual_end_ms: int | None = None
    wav: str | None = None

    def quantized_start(self) -> int:
        return round(self.start_ms / GRID_MS) * GRID_MS


@dataclass
class Timeline:
    scenario_id: str
    scenario_class: str
    events: list[Event] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(self, ev: Event) -> Event:
        self.events.append(ev)
        return ev

    def duration_ms(self) -> int:
        end = 0
        for ev in self.events:
            end = max(end, ev.actual_end_ms or 0, ev.quantized_start() + 2000)
        return end

    def overlap_stats(self) -> dict:
        speech = [e for e in self.events if e.kind != "marker" and e.text]
        overlap_ms = 0
        for i, a in enumerate(speech):
            for b in speech[i + 1:]:
                a0, a1 = a.actual_start_ms, a.actual_end_ms
                b0, b1 = b.actual_start_ms, b.actual_end_ms
                if None in (a0, a1, b0, b1):
                    continue
                overlap_ms += max(0, min(a1, b1) - max(a0, b0))
        return {"n_events": len(self.events), "overlap_ms": overlap_ms}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grid_ms"] = GRID_MS
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "Timeline":
        d = json.loads(Path(path).read_text())
        evs = [Event(**e) for e in d.pop("events")]
        d.pop("grid_ms", None)
        return cls(events=evs, **d)
