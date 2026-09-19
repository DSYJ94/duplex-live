# HANDOFF — where we are (2026-09-17)

Read this first in a fresh chat. Everything below is decided and documented.

## Current status

- **Runbook 00/01: DONE (Issue #1 closed).** Agentic SSH via
  `ssh u6si.aip2.isambard` (Clifton cert auto-renews on auth attempts; if
  `Permission denied (publickey)` just retry after a minute — cert window is
  12 h). ARM env verified on GH200; Qwen3-Omni thinker smoke passed. Env
  definition incl. all gotchas: `training/environment.md` — read before any
  GPU work.
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
- **Synthetic factory v0.2 (Issue #3, in progress — 2026-09-17 session).**
  v0.1 scale render (32 scenarios, 16.2 min) verified on Lustre
  (`runs/synth/`), then three v0.2 upgrades landed:
  1. **Quality filter** (`data/synthetic/filter.py`, `synth-filter.slurm`):
     rule checks (clipping, same-channel collisions, barge-in cut placement)
     + whisper-large-v3-turbo per-utterance round-trip WER (MIT, staged in
     hf-cache; runs under `pylibs`). Short refs (≤4 words: backchannels,
     spelled syllables) are duration-gated — ASR tokenization can't judge
     them. Old batch: 19/32 pass; all fails were generator bugs (below) or
     stale-audio artifacts.
  2. **Generator scheduling fix**: v0.1 placed same-channel starts from
     65 ms/char estimates; measured rate is 72–105 ms/char + lead-in.
     `_est_ms` (800 ms + 80 ms/char) now schedules all same-channel starts;
     local check: 0 collisions at per_class=8.
  3. **Voice pool** (`data/synthetic/voices.py`): 39 LibriSpeech
     test-clean speakers (CC-BY-4.0, staged at `$DL/models/voices/`) + 3
     MiniCPM anchors; `render.py --voices` assigns one distinct voice per
     channel, deterministic per scenario_id, `--anchor-prob 0.5`. Content
     banks expanded (15 explain / 10 stories / 12 hard questions / 8 timed
     topics / 9 barge pairs; paired barge/yield texts; timed-task requests
     match delivered cadence).
  **v0.2 VALIDATED (2026-09-17)**: batch `runs/synth-v02/` (32 scenarios,
  17.3 min, seed 41) → **28/32 pass (87.5%)**, 0 same-channel collisions
  (7 TTS overruns auto-repaired at mix time), voice pool validated (39
  LibriSpeech voices ≤ 0.074 weighted WER vs anchors 0.042–0.105, no
  pruning needed). Remaining fails = genuine defects the gate caught
  (backchannel TTS babble on 1–2-word prompts, marginal WER outliers) —
  attrition covered by over-render. **Throughput: ~0.7 GPU-h per audio-hour
  → 50 h ≈ 35 GPU-h; compute is not the bottleneck.**
  **Fidelity policies (2026-09-17, from user audition of banks-smoke):**
  (1) render order is cut → barge-yield push → same-channel repair → mix
  (repair-before-cut shipped 12–21 s reply gaps — user caught a 21,120 ms
  one); (2) **barge-yield policy**: assistant speech never starts during a
  realized `barge_in` — pushed to barge actual end +240 ms (human
  turn-taking offset; trains clean yield, never talk-over-interrupter);
  (3) filter gates `yield_latency` (>2500 ms) AND `yield_overlap`
  (<−160 ms) so neither failure ships unheard; (4) TIMED_TOPICS bank items
  must be numerically consistent (count nouns == callout count, "for N
  seconds" == span) — banks.json pruned 24→22 timed topics.
  **Audit by ear**: `python -m viz.listen` → http://localhost:8765 —
  serves locally synced batches (mixed/stems/per-utterance players joined
  with WER + whisper hypotheses) + voice pool; rsync recipes in its
  docstring; refresh after each sync.
  **ACCELERATED (2026-09-19, user call):** multi-worker GPU rendering
  (`--slice` shards + manifest merge + atomic cache publishes; chatterbox
  4 workers/GPU, minicpm 2) ≈3.5× batch throughput; **big7–10 queued in
  parallel** (seeds 707-710). Barge-in audition follow-up measured: pre-
  barge assistant silence median 0 ms / p90 0 ms across 96 scenarios in
  big5/2/1 — barges land mid-speech, not after predictive pauses; the
  perceived pause = prosodic dip/outlier, NOT a pipeline bug, arguably
  naturalistic. Layer-A (#7) starts EARLY (~Sep 22) instead of Sep 28 —
  all dependencies (tokenizer, factory data, eval gate, delegation brain)
  are at v0-done.
  **Hours-scale begun (2026-09-18): `synth-big1` = 256 scenarios, 2.2 h,
  1h24m GPU (0.85 GPU-h/audio-h), 182/256 pass (71%), WER median 0.018 /
  p90 0.111.** Factory total through big18: **28.12 h gate-passed / 3,673
  scenarios** (wave pass rates: chatterbox 86-89%, minicpm 78-82% — the
  clip-cache converged both engines).
  Failure profile is ONE engine limit: MiniCPM-o TTS babbles
  on 1–2-word prompts (tail_babble ×147; backchannel 32/32, timed_task
  29/32 fails). **Second engine staged and winning: Chatterbox
  (Apache-2.0)** — overlay `pylibs-chatterbox` (torch-stripped, weights in
  hf-cache, native 24 kHz), renderer refactored to pluggable engines
  (`--engine {minicpm,chatterbox}` in render.py, `ENGINE=` in
  synth-render.slurm; same voice pool + all mix-time policies + same
  filter). **`synth-big2` (Chatterbox, 256 scen., seed 303): 210/256
  (82%), WER median 0.000, timed_task 8 vs 29 fails, backchannel 23 vs 32;
  register-distinct voices active. Night 2026-09-18: big3 (chatterbox,
  206/256) + big4 (minicpm, 178/256) — engine rates stable (CB ~81%,
  MCPM ~70.5%). **FACTORY TOTAL: 5.81 h gate-passed / 776 scenarios
  (runs/synth-big{1..4}).** Clip-cache for ≤6-word renders LANDED, active
  from the next batch (`--clip-cache`, engine-scoped keys;
  `purge_cache.py` closes the loop — bad keys re-roll, passing clips
  freeze). Sesame CSM-1B
  fits the same abstraction later (mimi codec stack present in
  pylibs-pplex). **Render jobs self-chain their filter via `afterok`** —
  batches finish server-side through cert lapses; only the rsync-to-gallery
  step needs the agent.
  **Gate set added 2026-09-17/18 (from user auditions):** yield_latency
  >2500ms, yield_overlap <−160ms, insertion gate (tail_babble: ≥2 extra
  words anywhere, ≥1 on ≤6-word refs — caught a hallucinated "Doctor"),
  barge-yield policy pushes assistant speech behind realized barge end
  +240 ms (render-side, step 2.5, AFTER cut), user/assistant voices must
  differ in F0 register (voices_f0.json, pyin-measured).
- **Overlays**: `pylibs` (Qwen, transformers 5.17, +whisper judge),
  `pylibs-minicpm` (transformers 4.51 pinned, +accelerate 1.12),
  `pylibs-pplex` (moshi; sphn source-built — recipe in environment.md),
  `pylibs-mcpmdemo` (live demo).
- **Isambard workspace**: `/projects/u6si/duplex-live/` (`repo/` = git clone,
  read-write via deploy key; `models/hf-cache`, `models/voices/`, `runs/`,
  `logs/`, `tools/`). Login-node background processes are reaped when the
  ssh session exits — run long staging in the ssh FOREGROUND or sbatch it.
  SLURM `--output=logs/...` is relative to the SUBMISSION cwd (repo/ vs $DL/
  both appear in logs — check both).

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

1. Issue #3 — keep the factory accumulating (big5/big6 rendering; batches
   self-chain render→filter→purge; rsync + audition via viz.listen). Then
   Issue #5 — delegation boundary v0; ACP is the preferred shape
   (docs/notes-qwen-live.md) over a Kimi-specific adapter. Issue #6 — eval
   harness v0 (steal the frameHash/ACK diagnostics pattern from qwen-live).
2. Issue #7 — Layer-A on Qwen3-Omni + Phase-1 fine-tune (start ~Sep 28;
   `experts_implementation="batched_mm"` mandatory). Training sampler
   consumes tokenizer/ + factory train_manifests. Demo option: serve
   Layer-A behind a DashScope-realtime-compatible endpoint → qwen-live
   Host/daemon as free demo client (docs/notes-qwen-live.md).
3. Layer-2 alignment toolchain (CANDOR/Fisher/PMQs collector output →
   separation → 80 ms grid) once Layer-A training shape is stable.
4. If Qwen3.5+ Omni ships open weights: candidate Layer-A base swap
   (docs/notes-qwen-live.md watch item).

**Issue #5 v0 DONE (2026-09-19):** `serving/delegation/` — async boundary
(fire() 0.1 ms vs a 5 s brain; non-blocking poll(); pre-warmed session;
<3 s p50 first-chunk budget; ADR-0002 by construction). Backends: Mock
validated; Kimi (OpenAI-compatible SSE, needs key, base_url swappable to
self-hosted vLLM); ACP client spec-correct but live handshake pending
(local claude/codex builds predate --acp). Remaining: live Kimi run,
<delegate> interception in the Layer-A tick loop (#7).

**Issue #6 v0 DONE (2026-09-19):** `evals/harness/` — T-protocol replay
harness; reproduces every bake-off finding from the raw recordings
(MiniCPM: T3 pass / T2 5040ms / T5 14.6s engaged; PersonaPlex: T4 pass /
T5 7.5s; both T1 p50 >2s). Metrics run on the 80 ms grid — same cadence
as the factory gates; this is the Layer-A checkpoint gate. LIVE PATH
ADDED same day: MiniCPMDuplexTarget behind the Target seam (frameHash
chunk telemetry), `run.py --live minicpm` drives + scores in one command
(`evals-live.slurm`) — baselines and live runs share one metrics path.

**Issue #2 v0 DONE (2026-09-19):** `tokenizer/` — 80 ms micro-turn grid +
interleaved serializer + reserved vocab + SPEC.md contract; round-trip and
invariants verified over all 8 local batches (860 scenarios). Upgrades
queued: forced word alignment, BPE-level sampler tokens.

## Repo conventions

- Task board: GitHub Issues #1–#7.
- **Voice-sample audit loop (user request, 2026-09-17): anything that
  produces audio must be audible to the user via the localhost gallery.**
  After every render/filter/voice step: rsync the batch + any voice assets
  to local `runs/` (recipes in `viz/listen.py` docstring), make sure
  `python -m viz.listen` is running, and tell the user there is new audio
  at **http://localhost:8765** (they refresh; pages read the synced dirs
  live). The server is a detached nohup process — check
  `pgrep -f viz.listen` and restart on port 8765 if the Mac rebooted.
- `viz/index.html` is the living visualization — every decision updates it
  (viz/GOVERNANCE.md). Repo copy is canonical (local editable copy
  `duplex-live-viz/` does not exist yet).
- Local clone (OneDrive-synced): `~/Library/CloudStorage/OneDrive-Personal/Shared GitHub Repos/duplex-live`
- Isambard clone: `/projects/u6si/duplex-live/repo` — keep both pushed in sync.
