"""Fail-closed QUALITY RECORDS for training data (owner directive
2026-10-02: "make sure it never happens again where we proceed even when
the training data was bad... Make sure we dont go ahead when there are
failures in the conversations and the timing labels").

Plain words: this module turns a rendered batch's verification evidence
into one small file — quality.jsonl, one record per conversation — and
gives every training/eval job one function to call at startup. If a job
asks "may I train on this batch?" and the answer is not a documented YES
for every conversation it is about to load, the job STOPS with a plain
message instead of training on bad data.

What "verified" means (all of):
  1. every CLIP passed the per-clip judge — no gate failure (babble, tail
     babble, duration) and word error within PER_CLIP_WER_BAR for any clip
     ASR can judge (filter.clip_passes is the single definition);
  2. the TIMING answer key was audited against the real audio on the 80 ms
     grid (data.synthetic.timing_audit) and passed — timing matters to
     every consumer that reads emission grids, so require_timing=True is
     the default;
  3. the record exists, carries this schema, and covers every conversation
     the consumer is about to load.

Files (batch dir):
  quality.jsonl           one record per conversation (verified: true/false)
  quality_summary.json    counts + the rules used (schema, wer bar, timing)

Run:
  python -m data.synthetic.quality --batch runs/synth-scale1
  python -m data.synthetic.quality --batch runs/synth-scale1 \
      --timing runs/synth-scale1/timing_report.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "dl-quality-v1"
TIMING_VALUES = ("ok", "fail", "missing")


class DataNotVerified(RuntimeError):
    """Raised by assert_verified when a job is about to load lessons that
    have no passing quality record. Fail-closed: the job aborts."""


def verdict_from_report(report: dict, wer_bar: float) -> tuple[bool, list[str]]:
    """Fail-closed verdict for ONE conversation from its filter_report line.
    A conversation is verified only if EVERY clip passes the per-clip bar
    (filter.clip_passes) — the same rule the render-time repair loop applies."""
    from data.synthetic.filter import clip_passes

    reasons: list[str] = []
    clips = [u for u in report.get("asr_detail", {}).get("per_utterance", [])
             if u.get("kind") == "utterance" or "gate_fail" in u]
    for j, u in enumerate(clips):
        if clip_passes(u, wer_bar):
            continue
        why = u.get("gate_fail") or f"wer {u.get('wer')} > {wer_bar} bar"
        reasons.append(f"clip {j:02d}_{u.get('ch')}: {why}")
    return (not reasons), reasons


def stamp_batch(batch_dir: Path, wer_bar: float,
                timing_jsonl: Path | None = None) -> dict:
    """Write quality.jsonl (+ summary) for a rendered batch. Reads the
    factory's own filter_report.jsonl (every rendered conversation, passed
    or dropped) and, if given, the timing audit's timing_report.jsonl.
    Returns the summary dict."""
    from data.synthetic.filter import PER_CLIP_WER_BAR  # single source

    if wer_bar is None:
        wer_bar = PER_CLIP_WER_BAR
    batch_dir = Path(batch_dir)
    reports = [json.loads(l)
               for l in open(batch_dir / "filter_report.jsonl")]
    timing: dict[str, dict] = {}
    if timing_jsonl is not None:
        for l in open(timing_jsonl):
            r = json.loads(l)
            timing[r["scenario_id"]] = r

    n_ok = n_clip_ok = 0
    verified_by_id: dict[str, bool] = {}
    with open(batch_dir / "quality.jsonl", "w") as qf:
        for r in reports:
            ok, reasons = verdict_from_report(r, wer_bar)
            n_clip_ok += ok
            sid = r["scenario_id"]
            t = timing.get(sid, {})
            t_state = ("ok" if t.get("timing") == "ok"
                       else "fail" if t.get("timing") == "fail" else "missing")
            if t_state != "ok":
                ok = False
                if t_state == "fail":
                    reasons = reasons + [
                        f"timing audit FAIL: worst offset {t.get('max_offset_ms')}ms"
                        f" (bar {t.get('offset_bar_ms')}ms)"]
                else:
                    reasons = reasons + ["timing answer key not audited against "
                                         "real audio (run data.synthetic."
                                         "timing_audit)"]
            # AFTER the timing fold-in: this is the flag the manifest
            # alignment trusts (bug found 2026-10-03 — setting it before
            # kept clip-verified/timing-failed rows in train_manifest)
            verified_by_id[sid] = ok
            rec = {"schema": SCHEMA, "scenario_id": sid,
                   "scenario_class": r.get("scenario_class"),
                   "asr_judge": r.get("asr_model"),
                   "wer_bar": wer_bar,
                   "n_clips": len(r.get("asr_detail", {}).get("per_utterance", [])),
                   "timing": t_state,
                   "verified": ok, "reasons": reasons}
            qf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += ok
    summary = {"schema": SCHEMA, "batch": str(batch_dir),
               "wer_bar": wer_bar,
               "asr_judge": reports[0].get("asr_model") if reports else None,
               "n_conversations": len(reports),
               "n_clip_verified": n_clip_ok,
               "n_verified": n_ok,
               "timing_audited": bool(timing)}
    (batch_dir / "quality_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")

    # align the training manifest with the stamp: train_manifest.jsonl
    # becomes the VERIFIED rows only (the filter knows clips, not timing;
    # its 155-pass may exceed the stamp's verified count). Loud, backed
    # up, recorded — consumers keep reading train_manifest.jsonl and the
    # fail-closed aborts stay armed for true staleness.
    mpath = batch_dir / "train_manifest.jsonl"
    if mpath.is_file():
        rows = [json.loads(l) for l in open(mpath)]
        # dedup keeping first occurrence — recovery-pass renders APPEND
        # rows, so a lesson recovered on a later pass can appear twice
        # (found 2026-10-04: 4 duplicates; downstream whitelists are
        # unique so nothing was harmed, but counts must be true)
        seen: set[str] = set()
        uniq = [r for r in rows if not (r["scenario_id"] in seen
                                        or seen.add(r["scenario_id"]))]
        kept = [r for r in uniq
                if verified_by_id.get(r["scenario_id"], False)]
        n_dup = len(rows) - len(uniq)
        if len(kept) != len(rows):
            backup = batch_dir / "train_manifest.all.jsonl"
            if not backup.is_file():
                import shutil
                shutil.copyfile(mpath, backup)
            with open(mpath, "w") as mf:
                for r in kept:
                    mf.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[quality] train_manifest.jsonl aligned to the stamp: "
                  f"{len(kept)}/{len(rows)} rows kept"
                  + (f", {n_dup} duplicate(s) removed" if n_dup else "")
                  + f" (backup: train_manifest.all.jsonl) — dropped rows "
                  f"failed the per-clip bar or the timing audit")
    print(f"[quality] {batch_dir}: {n_ok}/{len(reports)} conversations "
          f"VERIFIED (per-clip bar {wer_bar}, timing "
          f"{'audited' if timing else 'NOT audited'}) — see quality.jsonl")
    return summary


def load_records(batch_dir: Path) -> dict[str, dict]:
    """scenario_id -> quality record. Raises DataNotVerified when the stamp
    is missing (nothing was verified) or carries a stale schema."""
    path = Path(batch_dir) / "quality.jsonl"
    if not path.is_file():
        raise DataNotVerified(
            f"NO QUALITY RECORD in {batch_dir} — the fail-closed data rule "
            f"stops here. Stamp the batch first: "
            f"python -m data.synthetic.quality --batch {batch_dir}")
    recs: dict[str, dict] = {}
    for l in open(path):
        r = json.loads(l)
        if r.get("schema") != SCHEMA:
            raise DataNotVerified(
                f"quality record in {batch_dir} has schema "
                f"{r.get('schema')!r}, this code requires {SCHEMA!r} — "
                f"re-stamp the batch")
        recs[r["scenario_id"]] = r
    return recs


def assert_verified(batch_dir: Path, role: str = "training",
                    require_timing: bool = True) -> dict[str, dict]:
    """The startup gate for every training/eval job: return quality records
    for the VERIFIED conversations of --batch, and RAISE (job aborts, plain
    message) if any conversation lacks one, fails the bar, or (default)
    lacks a passing timing audit. role is just for the error message."""
    batch_dir = Path(batch_dir)
    recs = load_records(batch_dir)
    verified = {sid: r for sid, r in recs.items() if r["verified"]}
    if require_timing:
        not_timed = [sid for sid, r in verified.items()
                     if r.get("timing") != "ok"]
        if not_timed:
            raise DataNotVerified(
                f"{len(not_timed)} verified conversation(s) in {batch_dir} "
                f"have NO PASSING TIMING AUDIT (first: {not_timed[0]}) — "
                f"the answer key's word timings were never checked against "
                f"the real audio. Run data.synthetic.timing_audit on the "
                f"batch, then re-stamp.")
    if not verified:
        raise DataNotVerified(
            f"0 of {len(recs)} conversations in {batch_dir} pass the "
            f"quality bar — {role} would proceed on bad data. Stopping.")
    timing_note = "timing ok" if require_timing else "timing not required"
    print(f"[quality] {batch_dir}: {len(verified)}/{len(recs)} conversations "
          f"verified ({timing_note}) — {role} may proceed")
    return verified


def checked_rows(batch_dir: Path, rows: list[dict], role: str = "training",
                 require_timing: bool = True) -> list[dict]:
    """Filter manifest rows to verified conversations, ABORTING (not
    silently dropping — the stale-shard lesson) if any row is not
    verified. Call right after reading train_manifest.jsonl."""
    recs = assert_verified(batch_dir, role=role,
                           require_timing=require_timing)
    bad = [r["scenario_id"] for r in rows if r["scenario_id"] not in recs]
    if bad:
        raise DataNotVerified(
            f"{len(bad)} conversation(s) about to be used by {role} have no "
            f"passing quality record (first: {bad[0]}) — fail-closed stop. "
            f"Re-render or repair them, then re-stamp the batch.")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True,
                    help="factory batch dir (filter_report.jsonl + "
                         "train_manifest.jsonl)")
    ap.add_argument("--wer-bar", type=float, default=None,
                    help="per-clip word-error bar (default: "
                         "filter.PER_CLIP_WER_BAR = 0.15)")
    ap.add_argument("--timing", default=None,
                    help="timing_report.jsonl from data.synthetic."
                         "timing_audit — without it every conversation "
                         "stamps timing=missing and verifies FALSE")
    args = ap.parse_args()
    stamp_batch(Path(args.batch), args.wer_bar,
                Path(args.timing) if args.timing else None)


if __name__ == "__main__":
    main()
