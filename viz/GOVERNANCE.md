# GOVERNANCE — the living visualization

`viz/index.html` is the **canonical living visualization** of the Duplex Live
plan. It exists so the architecture stays understandable beyond chat sessions.

## Rules

1. **Every architecture decision or scope change updates this page in the same
   commit (or immediately after).** If the plan changes and the page doesn't,
   the page is wrong — treat that as a bug.
2. The page mirrors the repo: candidates table ↔ ADR-0002; three-layer model ↔
   docs/08; tick-loop flow ↔ docs/09; roadmap status ↔ docs/07.
3. Bump the version note in the header (v1, v2, …) with a one-line changelog of
   what changed and why.
4. Keep it zero-dependency: single self-contained HTML file, no build step,
   opens in any browser.
5. The local editable copy lives in the workspace `duplex-live-viz/`; the repo
   copy under `viz/` is the durable record. Keep them in sync on every change.

## Changelog

- **v5 (2026-09-17):** bake-off closed (#4: both references fail addressee
  detection T5 — they are the behavioral bar, not the product); synthetic
  factory v0.2 (#3): 8 scenario classes rendered on the 80 ms grid with
  barge-in truncation at mix time, whisper-large-v3-turbo per-utterance
  quality gate (rule checks + WER), LibriSpeech CC-BY-4.0 voice pool +
  conservative same-channel scheduling. Header/banner refreshed to wk 1–2
  actuals.
- **v4 (2026-09-15):** bring-up complete — Issue #1 closed. ARM env verified
  (NGC 25.05 + pylibs overlay, `training/environment.md`); Qwen3-Omni-30B-A3B
  thinker smoke test passed on one GH200 (64.6 GB peak, MoE via `batched_mm`).
  PersonaPlex gated → awaiting HF token; MiniCPM-o bake-off unblocked.
- **v3 (2026-09-14):** allocation-deadline banner — 4,974 node-hours,
  terminates 2026-10-21; week-by-week execution plan; runbook pointers.
- **v2 (2026-09-14):** added §1½ three-layer duplex model (capability/policy/
  clock) + "the trick" (80 ms tick serialization); added MiniCPM-o 4.5 as fast
  path candidate; converted all timing from 200 ms micro-turns to the native
  80 ms AuT/Talker grid; added this governance rule.
- **v1 (2026-09-13):** initial candidate comparison + animated delegation flow.
