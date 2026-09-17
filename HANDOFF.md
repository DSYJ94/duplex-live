# HANDOFF — where we are (2026-09-16)

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
- **Live demos: DONE (2026-09-16), then stopped by user decision.** Both
  served bare-metal from SLURM jobs (`pplex-live.slurm`, `minicpm-live.slurm`
  — MiniCPM demo = OpenBMB 3-stage gateway/worker/backend ported to Apptainer,
  overlay `pylibs-mcpmdemo`, dep pins in the slurm script). Relaunch:
  `sbatch <script>` + `ssh -L <port>:<node>:<port> u6si.aip2.isambard`.
  **Live-test findings (user):** MiniCPM-o live = very slow/unresponsive
  (expected: PyTorch demo, no torch.compile, ~0.9–2 s/unit on A100-class,
  worse untuned); PersonaPlex live = genuinely duplex but hard to interrupt
  with the default persona (scripted T2 replay yielded ≤0.5 s). Conclusion:
  neither reference is a product — they are the behavioral bar + tooling.
- **Synthetic factory v0.1: BUILT + verified (Issue #3, in progress).**
  `data/synthetic/`: 80 ms grid Timeline, 8 scenario classes (hold_floor,
  addressee, backchannel, barge_in+retraction, timed_task, delegation,
  silence_semantics, simultaneous), MiniCPM-o zero-shot voice-cloned TTS
  renderer (3 voices) → mixed/stems/manifest on Lustre (`runs/synth/`).
  v0.1: barge-in truncation enforced at mix time. Scale render of 32
  scenarios (PER_CLASS=4) submitted 2026-09-17 (`synth-render.slurm`;
  check `squeue`/`logs/synth-render-*.out` before rerunning — wipe
  `runs/synth/` first, manifest appends). Next: model-based quality filter,
  wider voice pool, then hours-scale generation.
- **Overlays**: `pylibs` (Qwen, transformers 5.17), `pylibs-minicpm`
  (transformers 4.51 pinned, +accelerate 1.12), `pylibs-pplex` (moshi; sphn
  source-built — recipe in environment.md), `pylibs-mcpmdemo` (live demo).
- **Isambard workspace**: `/projects/u6si/duplex-live/` (`repo/` = git clone,
  read-write via deploy key; `models/hf-cache`, `runs/`, `logs/`, `tools/`).
  Login-node background processes are reaped — long jobs must be `sbatch`.

## The program

Build **Duplex Live**: a full-duplex, time-aware interaction model (GPT-Live /
TML interaction-model class) at mini open-source scale on Isambard-AI.

## Roles of the two references (clarified after live testing)

- **MiniCPM-o 4.5** (Apache-2.0): (a) reference implementation for our
  Layer-A tick loop (notes-minicpm-layera.md), (b) the factory's TTS voice
  engine (proven), (c) possible fast-path fine-tune base if Qwen3-Omni
  Layer-A slips. Its demo latency is a systems artifact, not the ceiling.
- **PersonaPlex 7B** (NVIDIA OML — eval/research only, NEVER training data):
  the interactivity bar + failure taxonomy source (T4 pass / T5 fail etc.).

Path to GPT-Live quality = our own model: Qwen3-Omni 30B + Layer-A spec
(docs/09) + synthetic-factory data + timing-aware evals (Issues #2/#3/#6/#7).
The references define the bar; they are not the product.

## Key decisions (ADRs in decisions/)

- **One product model:** Qwen3-Omni 30B-A3B (Apache-2.0) — fine-tune duplex in
  via the Layer-A spec (docs/09): 80 ms tick loop, interleaved causal stream,
  new `<sil>`/timing/`<delegate>`/`<yield>` embeddings. Backbone frozen Phase 1.
- **Brain:** Kimi API behind an async delegation boundary. Closed-API outputs
  never enter training data.
- **Duplex = 3 layers:** capability (architectural: tick loop), policy (data:
  synthetic factory), clock (systems: streaming session).

## Hard constraint

Isambard-AI Phase 2 allocation: **4,974 node-hours, terminates 2026-10-21**
(~5 weeks from 2026-09-14). Weeks 1–2: bring-up + bake-off + delegation v0.
Weeks 3–5: Layer-A + Phase-1 fine-tune → model v1.

## Next actions (in order)

1. Issue #3 — scale the factory: PER_CLASS≥4 renders, v0.1 barge-in cut at
   mix time, quality filter, then hours-scale generation.
2. Issue #2 — micro-turn tokenizer/streamer (80 ms grid; notes in
   docs/notes-minicpm-layera.md §1–3, §7). Gates grid-alignment of factory
   output and the training sampler.
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
