# HANDOFF — where we are (2026-09-15)

Read this first in a fresh chat. Everything below is decided and documented.

## Current status

- **Runbook 00/01: DONE.** Agentic SSH live via `ssh u6si.aip2.isambard`;
  ARM env verified on GH200; Qwen3-Omni-30B-A3B thinker smoke test passed
  (load 24.7s, 64.6 GB peak, correct generation). Env definition:
  `training/environment.md` — **read the two gotchas before any GPU work**
  (pylibs torch shadowing; MoE needs `experts_implementation="batched_mm"`).
- **Models staged** in HF cache on Lustre: Qwen3-Omni-30B-A3B, MiniCPM-o-4.5.
  PersonaPlex-7B-v1 gated — waiting on user to accept NVIDIA OML + place
  token at `/projects/u6si/duplex-live/.hf_token`.
- **Isambard workspace**: `/projects/u6si/duplex-live/` (`repo/` = git clone,
  read-write via deploy key; `models/hf-cache`, `pylibs/`, `runs/`, `logs/`).
  Container: `/projects/u6si/containers/latentsprites-pytorch-25.05.sif`.
- Login-node background processes are reaped on SSH logout — long jobs must
  be `sbatch` scripts (see `dl-models.slurm` in the workspace).

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
(~5 weeks from 2026-09-14). Weeks 1–2: bring-up + bake-off + delegation v0.
Weeks 3–5: Layer-A + Phase-1 fine-tune → model v1.

## Next actions (in order)

1. Runbook 02 — MiniCPM-o 4.5 duplex test (Issue #4 part 2; unblocked now) +
   code-reading notes → `docs/notes-minicpm-layera.md`.
2. PersonaPlex test (Issue #4 part 1) — starts once user places HF token.
3. Then Issue #5 (delegation v0) / #3 (synthetic factory v0) / #2 (micro-turn
   tokenizer) per the board.

## Repo conventions

- Task board: GitHub Issues #1–#7.
- `viz/index.html` is the living visualization — every decision updates it
  (viz/GOVERNANCE.md). Repo copy is canonical (local editable copy
  `duplex-live-viz/` does not exist yet).
- Local clone (OneDrive-synced): `~/Library/CloudStorage/OneDrive-Personal/Shared GitHub Repos/duplex-live`
- Isambard clone: `/projects/u6si/duplex-live/repo` — keep both pushed in sync.
