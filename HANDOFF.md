# HANDOFF — where we are (2026-09-14)

Read this first in a fresh chat. Everything below is decided and documented.

## The program

Build **Duplex Live**: a full-duplex, time-aware interaction model (GPT-Live /
TML interaction-model class) at mini open-source scale on Isambard-AI.

## Key decisions (ADRs in decisions/)

- **One product model:** Qwen3-Omni 30B-A3B (Apache-2.0) — fine-tune duplex in
  via the Layer-A spec (docs/09): 80 ms tick loop, interleaved causal stream,
  new `<sil>`/timing/`<delegate>`/`<yield>` embeddings. Backbone frozen Phase 1.
- **References, not products:** MiniCPM-o 4.5 (Apache-2.0, true duplex already —
  read its TDM/tick-loop code, possible fast-path demo) and PersonaPlex 7B
  (run + measure only; NVIDIA OML license).
- **Brain:** Kimi API behind an async delegation boundary. Closed-API outputs
  never enter training data.
- **Duplex = 3 layers:** capability (architectural: tick loop), policy (data:
  synthetic factory), clock (systems: streaming session).

## Hard constraint

Isambard-AI Phase 2 allocation: **4,974 node-hours, terminates 2026-10-21**
(~5 weeks). Weeks 1–2: bring-up + bake-off + delegation v0. Weeks 3–5:
Layer-A + Phase-1 fine-tune → model v1.

## Next actions (in order)

1. Runbook 00 — SSH key + ControlMaster + VS Code Remote-SSH; then say "go"
   and Kimi does the agentic bring-up (`ssh isambard`, env discovery).
2. Runbook 01 — Isambard bring-up (Issue #1).
3. Runbook 02 — PersonaPlex + MiniCPM-o bake-off (Issue #4); recordings become
   our first training data (docs/02).

## Repo conventions

- Task board: GitHub Issues #1–#7.
- `viz/index.html` is the living visualization — every decision updates it
  (viz/GOVERNANCE.md). Local editable copy: workspace `duplex-live-viz/`.
- Local clone (OneDrive-synced): `~/Library/CloudStorage/OneDrive-Personal/Shared GitHub Repos/duplex-live`
