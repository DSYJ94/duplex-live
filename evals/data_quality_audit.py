"""Batch-wide DATA-QUALITY AUDIT — plain-language numbers on how good a
rendered conversation batch actually is (2026-10-02, the fail-closed
data rule's measuring stick).

Plain words: this reads the factory's own per-clip speech-to-text
reports (filter_report.jsonl — every rendered conversation, passed or
not) and answers, with numbers:
  1. How many conversations were rendered, how many passed, what was
     dropped — per conversation class.
  2. WHY conversations were dropped: which check failed, and how many
     BAD CLIPS each dropped conversation contains (a conversation with
     exactly one bad clip is exactly what repair-in-render converts
     into a passing conversation — re-speak that one clip, keep the
     rest).
  3. How CLOSE the passing conversations are to the new, stricter
     per-clip bar: a lesson can pass today's aggregate gate while
     carrying clips that individually look bad (--wer-bar flags them).

Run (CPU, local or cluster):
  python -m evals.data_quality_audit --batch runs/synth-scale1
  python -m evals.data_quality_audit --batch runs/synth-scale1 --wer-bar 0.20
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_WER_BAR = 0.15  # stricter per-clip bar (today's gate has none;
# it only aggregates per channel at 0.30 and kills whole lessons)


def issue_kind(issue: str) -> str:
    """'tail_babble: 7 extra words (...)' -> 'tail_babble'."""
    return issue.split(":")[0].strip()


def clip_gate_fail(clip: dict, wer_bar: float) -> str | None:
    """Why would THIS clip fail the fail-closed bar? Returns a short
    tag or None. Uses the filter's own verdict first, then the audit
    bar for clips the filter's aggregate let through."""
    if clip.get("gate_fail"):
        return f"filter-{clip.get('gate', 'wer')}"
    if clip.get("kind") != "utterance":
        return None
    wer = clip.get("wer")
    if wer is None:
        return None
    if wer > wer_bar:
        return f"wer>{wer_bar:g}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True,
                    help="batch dir containing filter_report.jsonl and "
                         "train_manifest.jsonl")
    ap.add_argument("--wer-bar", type=float, default=DEFAULT_WER_BAR,
                    help="the per-clip word-error bar for the audit "
                         "(plain words: at most this fraction of words "
                         "mis-recognised per clip)")
    args = ap.parse_args()
    batch = Path(args.batch)

    reports = [json.loads(l)
               for l in open(batch / "filter_report.jsonl")]
    passed_ids = set()
    train_manifest = batch / "train_manifest.jsonl"
    if train_manifest.is_file():
        passed_ids = {json.loads(l)["scenario_id"]
                      for l in open(train_manifest)}
    else:
        passed_ids = {r["scenario_id"] for r in reports
                      if r["verdict"] == "pass"}

    print(f"[audit] batch {batch}: {len(reports)} conversations "
          f"rendered, {len(passed_ids)} passed the factory filter, "
          f"{len(reports) - len(passed_ids)} dropped")

    # ---- 1. per-class yield ----
    by_class: dict[str, Counter] = defaultdict(Counter)
    for r in reports:
        c = by_class[r["scenario_class"]]
        c["rendered"] += 1
        c["passed" if r["scenario_id"] in passed_ids else "dropped"] += 1
    for cls, c in sorted(by_class.items()):
        y = 100.0 * c["passed"] / max(c["rendered"], 1)
        print(f"[audit]   {cls}: {c['passed']}/{c['rendered']} passed "
              f"({y:.0f}% yield)")

    # ---- 2. why dropped: taxonomy + bad-clip counts ----
    taxonomy: Counter = Counter()
    bad_clips_per_dropped: Counter = Counter()
    for r in reports:
        if r["scenario_id"] in passed_ids:
            continue
        for issue in r["issues"]:
            taxonomy[issue_kind(issue)] += 1
        n_bad = sum(1 for u in r["asr_detail"]["per_utterance"]
                    if u.get("gate_fail"))
        bad_clips_per_dropped[n_bad] += 1
    print("[audit] drop reasons (issues can repeat per conversation):")
    for kind, n in taxonomy.most_common():
        print(f"[audit]   {kind}: {n}")
    print("[audit] gate-failing CLIPS per dropped conversation "
          "(1 bad clip = repair-in-render converts it):")
    for n_bad, n_lessons in sorted(bad_clips_per_dropped.items()):
        print(f"[audit]   {n_bad} bad clip(s): {n_lessons} conversations")

    # ---- 3. quality margin of PASSED conversations at the audit bar ----
    clip_wers: list[float] = []
    lessons_over_bar: list[tuple[str, int]] = []
    for r in reports:
        if r["scenario_id"] not in passed_ids:
            continue
        n_over = 0
        for u in r["asr_detail"]["per_utterance"]:
            if u.get("wer") is not None and u.get("kind") == "utterance":
                clip_wers.append(u["wer"])
                if clip_gate_fail(u, args.wer_bar):
                    n_over += 1
        if n_over:
            lessons_over_bar.append((r["scenario_id"], n_over))
    clip_wers.sort()
    n = len(clip_wers)
    if n:
        p50 = clip_wers[n // 2]
        p95 = clip_wers[min(n - 1, int(n * 0.95))]
        p99 = clip_wers[min(n - 1, int(n * 0.99))]
        print(f"[audit] PASSED conversations, per-clip word error: "
              f"p50 {p50:.3f}, p95 {p95:.3f}, p99 {p99:.3f}, "
              f"max {clip_wers[-1]:.3f} over {n} clips")
    print(f"[audit] passed conversations carrying clips ABOVE the "
          f"{args.wer_bar:g} per-clip bar: {len(lessons_over_bar)} "
          f"(these pass today's aggregate gate but fail the "
          f"fail-closed rule)")
    for sid, n_over in lessons_over_bar[:12]:
        print(f"[audit]   {sid}: {n_over} clip(s) over bar")
    if len(lessons_over_bar) > 12:
        print(f"[audit]   ... and {len(lessons_over_bar) - 12} more")

    # ---- 4. verdict ----
    n_one_bad = bad_clips_per_dropped.get(1, 0)
    n_dropped = len(reports) - len(passed_ids)
    print(f"[audit] PLAIN WORDS: if every dropped conversation with "
          f"exactly 1 bad clip is repaired by re-speaking that clip "
          f"(repair-in-render), up to {n_one_bad} of {n_dropped} "
          f"dropped conversations come back — yield would rise from "
          f"{len(passed_ids)}/{len(reports)} toward "
          f"{len(passed_ids) + n_one_bad}/{len(reports)}. The "
          f"{args.wer_bar:g}-bar check on PASSED conversations is the "
          f"fail-closed rule's second net: {len(lessons_over_bar)} "
          f"of {len(passed_ids)} would need clip repair too.")


if __name__ == "__main__":
    main()
