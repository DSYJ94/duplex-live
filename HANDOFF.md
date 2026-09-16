# HANDOFF — where we are (2026-09-15)

Read this first in a fresh chat. Everything below is decided and documented.

## Current status

- **Runbook 00/01: DONE (Issue #1 closed).** Agentic SSH via
  `ssh u6si.aip2.isambard` (Clifton cert — re-auth when it lapses); ARM env
  verified on GH200; Qwen3-Omni thinker smoke passed. Env definition incl.
  all gotchas: `training/environment.md` — read before any GPU work.
- **Runbook 02: DONE (Issue #4 closed).** Both references measured on one
  scripted T1–T5 protocol (`evals/baselines/`): PersonaPlex passes the 3 s
  thinking pause, MiniCPM-o passes backchannels; **both fail addressee
  detection (T5)** → factory target. Layer-A code notes:
  `docs/notes-minicpm-layera.md`. Recordings on Lustre (`runs/bakeoff*`) =
  data seed.
- **Overlays**: `pylibs` (Qwen, transformers 5.17), `pylibs-minicpm`
  (transformers 4.51 pinned), `pylibs-pplex` (moshi; sphn source-built —
  recipe in environment.md + `pplex-build.slurm`).
- **Isambard workspace**: `/projects/u6si/duplex-live/` (`repo/` = git clone,
  read-write via deploy key; `models/hf-cache`, `runs/`, `logs/`, `tools/`).
  Login-node background processes are reaped — long jobs must be `sbatch`.

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

1. Issue #3 — synthetic interaction-dynamics factory v0. Required scenario
   classes from the bake-off: hold-floor during thinking pauses (T4),
   addressee discrimination (T5), backchannel tolerance (T3), barge-in
   yield (T2). PersonaPlex/Qwen3-Omni talker as data voices.
2. Issue #2 — micro-turn tokenizer/streamer (80 ms grid; notes in
   docs/notes-minicpm-layera.md §1–3, §7).
3. Issue #5 — delegation boundary v0 (Kimi API) and #6 — eval harness v0
   (seed: `evals/bakeoff/minicpm_bakeoff.py` + per-chunk telemetry).
4. Week 3+: Issue #7 — Layer-A on Qwen3-Omni + Phase-1 fine-tune
   (`experts_implementation="batched_mm"` mandatory).

## Repo conventions

- Task board: GitHub Issues #1–#7.
- `viz/index.html` is the living visualization — every decision updates it
  (viz/GOVERNANCE.md). Repo copy is canonical (local editable copy
  `duplex-live-viz/` does not exist yet).
- Local clone (OneDrive-synced): `~/Library/CloudStorage/OneDrive-Personal/Shared GitHub Repos/duplex-live`
- Isambard clone: `/projects/u6si/duplex-live/repo` — keep both pushed in sync.
