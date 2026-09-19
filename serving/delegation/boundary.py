"""The async delegation boundary (Issue #5, docs/01 'brain').

Contract:
- The media/tick loop NEVER blocks on a delegation: `fire()` returns a
  handle in microseconds; chunks land on a queue the loop drains between
  ticks.
- One pre-warmed background session per conversation (context sent once,
  prompt-cache friendly, session affinity).
- Latency budget tracked per delegation: t_fire -> t_first_chunk (the
  <3 s p50 metric) -> t_done.
- Provenance: every request/response row is ledgered with closed_api
  flags. Ledger + outputs live under runs/delegation/ ONLY — never under
  data/ or any training manifest (ADR-0002 enforcement by construction).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

BUDGET_FIRST_CHUNK_S = 3.0


class Delegation:
    """One in-flight delegated question."""

    def __init__(self, question: str, t_fire: float):
        self.question = question
        self.t_fire = t_fire
        self.t_first_chunk: float | None = None
        self.t_done: float | None = None
        self.chunks: list[str] = []
        self.error: str | None = None
        self.done_evt = threading.Event()

    @property
    def first_chunk_s(self) -> float | None:
        return None if self.t_first_chunk is None else self.t_first_chunk - self.t_fire

    @property
    def total_s(self) -> float | None:
        return None if self.t_done is None else self.t_done - self.t_fire


class DelegationBoundary:
    def __init__(self, backend, ledger_dir: Path | str = "runs/delegation",
                 stall_phrases: list[str] | None = None):
        self.backend = backend
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.stall_phrases = stall_phrases or [
            "Good question, let me think that through properly.",
            "That deserves a careful answer — one moment.",
        ]
        self.closed_api = backend.name != "mock"
        self._q: queue.Queue = queue.Queue()
        self._ledger = open(self.ledger_dir / "delegations.jsonl", "a")
        self._stall_i = 0
        self.prewarm_s: float | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self, context: str) -> float:
        """Pre-warm the background session at conversation start."""
        self.prewarm_s = self.backend.prewarm(context)
        self._log({"event": "prewarm", "backend": self.backend.name,
                   "context_chars": len(context),
                   "prewarm_s": round(self.prewarm_s, 3)})
        return self.prewarm_s

    # ------------------------------------------------------------ the seam
    def fire(self, question: str) -> Delegation:
        """Non-blocking: spawn the ask on a worker thread, return now."""
        d = Delegation(question, time.time())
        threading.Thread(target=self._worker, args=(d,), daemon=True).start()
        return d

    def _worker(self, d: Delegation) -> None:
        try:
            for chunk in self.backend.ask_stream(d.question):
                if d.t_first_chunk is None:
                    d.t_first_chunk = time.time()
                d.chunks.append(chunk)
                self._q.put((d, chunk))
            d.t_done = time.time()
        except Exception as e:  # noqa: BLE001 — surfaced to the caller
            d.error = f"{type(e).__name__}: {e}"
            d.t_done = time.time()
        finally:
            d.done_evt.set()
            self._q.put((d, None))
            self._log({"event": "delegation", "backend": self.backend.name,
                       "closed_api": self.closed_api,
                       "question": d.question[:200],
                       "answer_chars": sum(map(len, d.chunks)),
                       "first_chunk_s": _r(d.first_chunk_s),
                       "total_s": _r(d.total_s), "error": d.error})

    # ------------------------------------------------------------ consumer
    def poll(self, timeout_s: float = 0.0):
        """Drain one event between ticks — the media loop's only duty."""
        try:
            return self._q.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def stall_phrase(self) -> str:
        """Rotate graceful stall lines spoken while a delegation runs."""
        phrase = self.stall_phrases[self._stall_i % len(self.stall_phrases)]
        self._stall_i += 1
        return phrase

    def _log(self, rec: dict) -> None:
        rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._ledger.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._ledger.flush()

    def budget_report(self, delegations: list[Delegation]) -> dict:
        fcs = sorted(d.first_chunk_s for d in delegations
                     if d.first_chunk_s is not None)
        p50 = fcs[len(fcs) // 2] if fcs else None
        return {"n": len(delegations),
                "first_chunk_p50_s": _r(p50),
                "within_budget": (p50 is not None and p50 <= BUDGET_FIRST_CHUNK_S),
                "budget_s": BUDGET_FIRST_CHUNK_S}


def _r(x):
    return round(x, 3) if x is not None else None
