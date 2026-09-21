"""Eval harness v0 runner (Issue #6).

Offline replay today (bake-off recordings on disk), live targets next:
the Target interface is the seam — ReplayTarget now, MiniCPMDuplexTarget
and LayerATarget (Issue #7) reuse the same protocol + metrics + report.

  python -m evals.harness.run --replay runs/bakeoff          # MiniCPM
  python -m evals.harness.run --replay runs/bakeoff-pplex    # PersonaPlex

Reads <replay>/<test>/{user_in.wav,model_out.wav,log.jsonl}; writes
<replay>/harness_report.json + prints a verdict table.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.harness import metrics            # noqa: E402
from evals.harness.protocol import T_PROTOCOL, THRESH  # noqa: E402


def load_wav(p: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def verdict(test: str, m: dict) -> str:
    if test == "t2":
        r = m.get("reply_after_barge_ms")
        if r is None:
            return "FAIL (never replied after barge)"
        ok = (r <= THRESH["reply_after_barge_max_ms"]
              and not m.get("interrupted_answer", False))
        return "PASS" if ok else f"FAIL (reply {r} ms > {THRESH['reply_after_barge_max_ms']})" \
            if r > THRESH["reply_after_barge_max_ms"] else "FAIL (answer broke early)"
    if test == "t3":
        n = m.get("backchannel_yields", 99)
        return "PASS" if n <= THRESH["backchannel_yield_max_count"] \
            else f"FAIL (yielded on {n} backchannels)"
    if test == "t4":
        return "PASS" if m.get("answered_part2") and not m.get("premature_answer") \
            else "FAIL (" + ("no part2 answer" if not m.get("answered_part2")
                             else "answered before user finished") + ")"
    if test == "t5":
        s = m.get("false_engagement_s", 99)
        return "PASS" if s <= THRESH["false_engagement_max_s"] \
            else f"FAIL (engaged {s}s with background speech)"
    if test == "t1":
        p50 = m.get("response_latency_p50_ms")
        if p50 is None:
            return "FAIL (no answers)"
        return "PASS" if p50 <= THRESH["response_latency_p50_max_ms"] \
            else f"FAIL (p50 {p50} ms)"
    return "n/a"


def replay(root: Path, tests: list[str]) -> list[dict]:
    rows = []
    for t in tests:
        d = root / t
        spec = T_PROTOCOL[t]
        out, sr = load_wav(d / "model_out.wav")
        inp = None
        if (d / "user_in.wav").exists():
            inp, _ = load_wav(d / "user_in.wav")
        m = metrics.measure(t, spec["marks"], out, sr, inp)
        m["verdict"] = verdict(t, m)
        # telemetry aggregates when the target logged chunks (MiniCPM)
        logp = d / "log.jsonl"
        if logp.exists():
            recs = [json.loads(l) for l in logp.read_text().splitlines() if l.strip()]
            walls = [r["wall_s"] for r in recs if "wall_s" in r]
            listens = [r.get("is_listen") for r in recs]
            m["chunks"] = len(recs)
            m["wall_p50_s"] = round(statistics.median(walls), 3) if walls else None
            m["listen_fraction"] = round(
                sum(1 for x in listens if x) / len(listens), 3) if listens else None
        rows.append(m)
        print(f"[{t}] {spec['intent']}")
        for k, v in m.items():
            if k not in ("test",):
                print(f"    {k}: {v}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="recordings root (offline metrics)")
    ap.add_argument("--live", default=None,
                    help="'minicpm' or 'layera' — drive a live target, "
                         "then score it")
    ap.add_argument("--ckpt", default=None,
                    help="layera Phase-1 checkpoint (runs/p1-vX/p1_checkpoint.pt)")
    ap.add_argument("--out", default=None, help="live output root")
    ap.add_argument("--prompts", default=None,
                    help="stimulus prompts dir (bake-off Phase A layout)")
    ap.add_argument("--assets", default=None, help="MiniCPM snapshot assets")
    ap.add_argument("--tests", default="t1,t2,t3,t4,t5")
    args = ap.parse_args()
    tests = [t for t in args.tests.split(",") if t]

    if args.live == "minicpm":
        from evals.harness.protocol import LIVE_TESTS
        from evals.harness.targets import MiniCPMDuplexTarget, run_live

        out_root = Path(args.out or "runs/harness-live")
        prompts = Path(args.prompts or "runs/bakeoff/prompts")
        assets = Path(args.assets)
        tgt = MiniCPMDuplexTarget()
        tgt.prepare(assets=assets)
        run_live(tgt, {t: LIVE_TESTS[t] for t in tests}, out_root, prompts)
        args.replay = str(out_root)

    if args.live == "layera":
        from evals.harness.protocol import LIVE_TESTS
        from evals.harness.targets import LayerATarget, run_live

        if not args.ckpt:
            ap.error("--live layera requires --ckpt")
        out_root = Path(args.out or "runs/harness-layera")
        prompts = Path(args.prompts or "runs/bakeoff/prompts")
        tgt = LayerATarget()
        tgt.prepare(ckpt=args.ckpt)
        run_live(tgt, {t: LIVE_TESTS[t] for t in tests}, out_root, prompts)
        args.replay = str(out_root)

    if not args.replay:
        ap.error("--replay or --live required")
    root = Path(args.replay)
    rows = replay(root, tests)

    outp = root / "harness_report.json"
    outp.write_text(json.dumps({"target": str(root), "rows": rows}, indent=1))
    print(f"\n{'test':5s} {'verdict':48s} headline")
    for m in rows:
        head = (m.get("reply_after_barge_ms") or m.get("backchannel_yields")
                or m.get("false_engagement_s") or m.get("response_latency_p50_ms")
                or m.get("answered_part2"))
        print(f"{m['test']:5s} {m['verdict']:48s} {head}")
    print(f"report: {outp}")


if __name__ == "__main__":
    main()
