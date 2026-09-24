"""Word/silence density audit over a rendered batch's 80 ms grids.

Plain words: after the word-dense lessons rendered, this counts what
fraction of ticks carry a word, <sil>, a timing token, or a yield —
the number that decides whether the poles can learn at equal rates.

Cluster usage (CPU-only, no GPU):
  apptainer exec --bind /projects \
    --env PYTHONPATH=$DL/pylibs:$DL/repo $CONTAINER \
    python -m evals.grid_density_audit --batch $DL/runs/synth-dense1/audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True,
                    help="batch dir containing train_manifest.jsonl")
    args = ap.parse_args()
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tokenizer.grid import from_scenario_dir

    batch = Path(args.batch)
    rows = [json.loads(l) for l in open(batch / "train_manifest.jsonl")]
    hdr = f"{'scenario':<18s} {'ticks':>6s} {'word':>6s} {'sil':>6s} {'t+tok':>7s} {'yield':>6s}"
    print(hdr)
    tot = dict(w=0, s=0, t=0, n=0)
    for r in rows:
        g = from_scenario_dir(batch / r["dir"])
        w = s = t = y = 0
        for tick in g.ticks:
            e = tick.emission
            if e == "<sil>":
                s += 1
            elif e.startswith("<t+"):
                t += 1
            elif e.startswith("<"):
                y += e == "<yield>"
            else:
                w += 1
        n = g.n_ticks
        tot["w"] += w; tot["s"] += s; tot["t"] += t; tot["n"] += n
        print(f"{g.scenario_id:<18s} {n:6d} {w / n:6.0%} {s / n:6.0%} "
              f"{t / n:7.1%} {y:6d}")
    n = tot["n"]
    if n:
        print(f"{'TOTAL':<18s} {n:6d} {tot['w'] / n:6.0%} "
              f"{tot['s'] / n:6.0%} {tot['t'] / n:7.1%}")


if __name__ == "__main__":
    main()
