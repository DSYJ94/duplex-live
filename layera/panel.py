"""Regression-gate panel (2026-10-01): the side-effect eval that gates
every RL-era readout.

Principle (owner-approved mitigation set): the RL reward cannot see the
side effects, so the eval must — and it must GATE, not just report. A
checkpoint that improves the reward while regressing placement,
completeness, or quiet discipline is rejected, reward gain irrelevant.

This parses freerun_diag.py's stdout log(s) — no re-running, works
retroactively on every past and future freerun (probe-1, GRPO sweeps,
scale1 baselines) — and applies the panel:

  per lesson:  exact-onset fraction, +-1s fraction (v2/v3 word-onsets
               line), v3 false-speech count, v3 speech recall, full-mute
               flag (exact=0 AND chatter=0 AND R3~0 = the mute pole),
  gates:       paired comparison against a baseline log (same lesson,
               same occurrence index): FAIL when a candidate drops
               exact or +-1s placement beyond tolerance, raises chatter
               beyond tolerance, or trips the mute flag.

One log line of freerun output per occurrence is kept IN ORDER; a sweep
log (thr=0.35, 0.5, 0.65 on the same lessons) pairs occurrence k of the
candidate with occurrence k of the baseline — pass the SAME shape of
log on both sides.

  python -m layera.panel --log run.log                    # report only
  python -m layera.panel --log run.log --gate-vs base.log # gate it

Exit code 0 = PASS, 1 = FAIL (usable from slurm/shell).
"""

from __future__ import annotations

import argparse
import re
import sys

RE_LESSON = re.compile(r"^\[freerun\] (\S+): \d+ ticks")
RE_V3 = re.compile(
    r"SPEECH v3 \(speak-gold=speech\): P ([\d.]+)  R ([\d.]+)  "
    r"F1 ([\d.]+); false speech on (\d+) genuinely-quiet ticks")
RE_ONSET = re.compile(
    r"WORD ONSETS: (\d+)/(\d+) exact, (\d+)/(\d+) within \+-1s; "
    r"(\d+) continuation pieces")
RE_MUTE = "decode gates"


def parse_log(path: str) -> list[dict]:
    """All COMPLETE freerun occurrences in one log, in file order.
    An occurrence completes with its WORD ONSETS line (last metric of
    a lesson in freerun_diag); crash-truncated lesson headers drop out
    naturally at the final filter."""
    rows, cur = [], None
    for line in open(path, errors="replace"):
        m = RE_LESSON.match(line)
        if m:
            cur = {"sid": m.group(1)}
            rows.append(cur)
            continue
        if cur is None:
            continue
        m = RE_V3.search(line)
        if m:
            cur["r3"], cur["chatter3"] = float(m.group(2)), int(m.group(4))
            continue
        m = RE_ONSET.search(line)
        if m:
            cur["exact"], cur["n_onset"] = int(m.group(1)), int(m.group(2))
            cur["near"], cur["pieces"] = int(m.group(3)), int(m.group(5))
            cur["exact_frac"] = cur["exact"] / max(1, cur["n_onset"])
            cur["near_frac"] = cur["near"] / max(1, cur["n_onset"])
            cur["mute"] = (cur["exact"] == 0 and cur["chatter3"] == 0
                           and cur["r3"] < 0.02)
    return [r for r in rows if "exact_frac" in r]


def show(rows: list[dict], title: str) -> None:
    print(f"\n== {title}: {len(rows)} freerun occurrences ==")
    print(f"{'lesson':22s} {'#':>2s} {'exact':>9s} {'+-1s':>9s} "
          f"{'chatter3':>8s} {'R3':>5s} {'pcs':>4s}  flags")
    for i, r in enumerate(rows):
        flags = ["FULL-MUTE"] if r.get("mute") else []
        print(f"{r['sid']:22s} {i:>2d} {r['exact']:>4d}/{r['n_onset']:<4d} "
              f"{r['near']:>4d}/{r['n_onset']:<4d} {r['chatter3']:>8d} "
              f"{r['r3']:>5.2f} {r['pieces']:>4d}  {' '.join(flags)}")
    n = sum(r["n_onset"] for r in rows)
    if n:
        ex = sum(r["exact"] for r in rows) / n
        nr = sum(r["near"] for r in rows) / n
        print(f"{'AGGREGATE':22s}    exact {ex:5.1%}   within +-1s "
              f"{nr:5.1%}   (bar v2 needs >=90% exact)")


def gate(cand: list[dict], base: list[dict], tol_exact: float,
         tol_near: float, tol_chatter: int) -> bool:
    """Paired gate: occurrence k of a lesson vs occurrence k in the
    baseline. Any regression beyond tolerance, or a full-mute flag,
    fails the candidate. Returns True = PASS."""
    by = {}
    for r in base:
        by.setdefault(r["sid"], []).append(r)
    fails, unpaired = [], []
    seen, kth = set(), {}
    for r in cand:
        kth[r["sid"]] = kth.get(r["sid"], 0) + 1
        k = kth[r["sid"]]
        pool = by.get(r["sid"], [])
        b = pool[k - 1] if len(pool) >= k else None
        if b is None:
            unpaired.append(f"{r['sid']}#{k}")
            continue
        seen.add((r["sid"], k))
        why = []
        if r["exact_frac"] < b["exact_frac"] - tol_exact:
            why.append(f"exact {r['exact_frac']:.0%} < "
                       f"{b['exact_frac']:.0%} -{tol_exact:.0%}")
        if r["near_frac"] < b["near_frac"] - tol_near:
            why.append(f"+-1s {r['near_frac']:.0%} < "
                       f"{b['near_frac']:.0%} -{tol_near:.0%}")
        if r["chatter3"] > b["chatter3"] + tol_chatter:
            why.append(f"chatter {r['chatter3']} > "
                       f"{b['chatter3']} +{tol_chatter}")
        # regression-gate semantics: only a NEW mute fails the
        # candidate. Baseline-mute lessons (e.g. barge-114 in scale1)
        # were dead before training; their persistence is the bar's
        # problem, not a side effect. A recovered mute is an
        # improvement worth naming on the PASS line.
        recovered = b.get("mute") and not r.get("mute")
        if r.get("mute") and not b.get("mute"):
            why.append("NEW full-mute pole")
        tag = f"{r['sid']}#{k}"
        verdict = "PASS" if not why else "FAIL — " + "; ".join(why)
        if recovered and not why:
            verdict += " (recovered from baseline mute)"
        print(f"  [gate] {tag:24s} {verdict}")
        if why:
            fails.append(tag)
    for sid, occs in by.items():
        for k in range(1, len(occs) + 1):
            if (sid, k) not in seen:
                unpaired.append(f"{sid}#{k} (baseline only)")
    if unpaired:
        print(f"  [gate] unpaired (report only): {', '.join(unpaired)}")
    print(f"  [gate] VERDICT: {'PASS' if not fails else 'FAIL'} "
          f"({len(fails)} lesson-gate(s) tripped)")
    return not fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", required=True,
                    help="freerun_diag log to report (repeatable)")
    ap.add_argument("--gate-vs", default=None,
                    help="baseline log for the paired regression gate")
    ap.add_argument("--tol-exact", type=float, default=0.05,
                    help="max exact-fraction DROP vs baseline (default 5pt)")
    ap.add_argument("--tol-near", type=float, default=0.05,
                    help="max +-1s-fraction DROP vs baseline")
    ap.add_argument("--tol-chatter", type=int, default=50,
                    help="max chatter-tick RISE vs baseline")
    args = ap.parse_args()

    cand: list[dict] = []
    for p in args.log:
        rows = parse_log(p)
        show(rows, p)
        cand += rows
    if not args.gate_vs:
        return
    base = parse_log(args.gate_vs)
    show(base, f"baseline: {args.gate_vs}")
    ok = gate(cand, base, args.tol_exact, args.tol_near, args.tol_chatter)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
