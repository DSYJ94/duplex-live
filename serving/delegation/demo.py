"""Delegation boundary demo / smoke test (Issue #5).

Runs the full shape of a delegated turn against any backend and reports
the latency budget:

  stall phrase spoken -> fire() (never blocks) -> chunks stream back on
  poll() -> woven result -> budget report

  python -m serving.delegation.demo --backend mock
  python -m serving.delegation.demo --backend acp -- claude --acp
  KIMI_API_KEY=sk-… python -m serving.delegation.demo --backend kimi
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serving.delegation.backends import AcpBackend, KimiBackend, MockBackend  # noqa: E402
from serving.delegation.boundary import DelegationBoundary              # noqa: E402

QUESTION = ("Can you compare the consistency guarantees of Raft and Paxos, "
            "and tell me which one I'd want for a globally distributed "
            "ledger and why?")
CONTEXT = ("You are the background reasoning brain behind a duplex voice "
           "assistant. Answer concisely, headline first.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "acp", "kimi"])
    ap.add_argument("--ledger", default="runs/delegation")
    ap.add_argument("acp_command", nargs="*", help="acp only: agent command")
    args = ap.parse_args()

    if args.backend == "mock":
        backend = MockBackend(first_chunk_s=0.8)
    elif args.backend == "kimi":
        backend = KimiBackend()
    else:
        cmd = args.acp_command or ["claude", "--acp"]
        backend = AcpBackend(cmd)

    b = DelegationBoundary(backend, ledger_dir=args.ledger)
    warm = b.start(CONTEXT)
    print(f"[prewarm] {backend.name}: {warm:.2f}s (closed_api={b.closed_api})")

    print(f"[stall]   assistant says: {b.stall_phrase()!r}")
    t0 = time.time()
    d = b.fire(QUESTION)
    fire_us = (time.time() - t0) * 1e6
    print(f"[fire]    returned in {fire_us:.0f} µs (non-blocking ✓)")

    woven = []
    while True:
        ev = b.poll(timeout_s=1.0)
        if ev is None:
            continue
        dlg, chunk = ev
        if chunk is None:
            break
        if not woven:
            print(f"[weave]   first chunk after {dlg.first_chunk_s:.2f}s")
        woven.append(chunk)
        print(f"  … {chunk[:70]!r}")
    print(f"[done]    {d.total_s:.2f}s total, {sum(map(len, woven))} chars, "
          f"error={d.error}")
    print(f"[budget]  {b.budget_report([d])}")
    print(f"[ledger]  {Path(args.ledger)}/delegations.jsonl "
          f"(closed_api rows never enter training data)")


if __name__ == "__main__":
    main()
