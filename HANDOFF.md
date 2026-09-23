# HANDOFF — where we are (2026-09-23)

Read this first in a fresh chat. Everything below is decided and documented.

## The frontier right now (2026-09-23)

- **Phase-1 training is a dose-response study through v11.** The ablation
  ledger with every recipe, exam number and verdict is **EXPERIMENTS.md** —
  read it before proposing any new run. Training is ~1.1 GPU-h per dose
  (55 min train + 10 min held-out exam), and **the exam predicts the live
  gate** (validated on v10: its exam word-slot sil-steals mapped exactly to
  live muting), so the exam is the default decision instrument; rerun the
  T1–T5 gate only when the exam moves.
- **All three live-gated checkpoints (v2 unweighted, v3 w50, v10 w50+tax)
  pass exactly 2/5**, and the shared failure is UNDER-SPEAKING: silence is
  ~95% of training ticks, so every model learns a silence attractor — even
  unweighted v2 mutes on t1 (0 word beats). v3 is still the only checkpoint
  that fires a countdown (t+30s, 50% exam) and speaks live; v10 fixed the
  rare-token thief (1–2/lesson) at the cost of all timing and all speech.
- **In flight: v11, the speak-up dose** (user-approved 2026-09-21):
  w50 + thief-pen 2 + `--word-weight 3`. **RESULT (2026-09-22): the exam
  moved decisively — words 0.185–0.387 (v2-level restored), t+30s
  partially back (1/4, prob 0.426), barge-class silence kept; cost: the
  t+30s thief re-empowered on the countdown-dense lesson (251 steals).
  Live gate: PASS 2/5, beat census v2-shaped (sparse, misplaced speech;
  mute on t1/t2). Exam tracks the AMOUNT of speaking, not placement.**
- **External adversarial review landed (CODEX_PLAN.md, 2026-09-21) and
  VERIFIED accurate; the owner APPROVED the 5-point adoption plan
  (2026-09-22). Executed:** (1) ledger corrections committed (v5/v6
  exposure, freerun 159-steals, rounding); (2) **gate instruments
  repaired** — marks aligned to true stimulus positions, latencies
  anchored at utterance END, unanswered questions counted (t1 needs
  ≥8/10 answered), t2 reply from barge END, t4 talk-over = premature,
  `<sil>`-run (not single tick) ends a turn, 16 kHz/24 kHz framing bug
  fixed — all guarded by `evals/harness/test_gate_regressions.py`
  (three-tone cheat now FAILS, honest traces PASS, all green);
  (3) **decode knobs live**: `--sil-bias` + word continuation
  (`[pad w1 w2]`, the training contract for 33.6% multi-piece words)
  in `layera/duplex.py`, plumbed through the harness;
  `freerun_diag.py` rewritten free-mode-only with the same knobs;
  `layera/contract_trace.py` prints the training-vs-live token
  streams. **All five points EXECUTED and answered (2026-09-22):**
  rescored verdict table (v2/v3/v10/v11 under honest instruments —
  the old "2/5 each" was composition noise), decode sweep (monotone
  dial: bias buys onset exactness with chatter; held-back lesson 017
  unlocked 0/19→19/19), and the live gate at bias 0/4: **v11's live
  muting was substantially a decode threshold** — bias 4 takes
  question-end answers 0/10→5/10 and produces the first post-barge
  reply (1920 ms), with the predicted costs (background chatter,
  one yield). Decode is a diagnosis knob, not a fix: latency (3.1 s
  vs 1.5 s), part-2 handback, and amount/control coupling are
  training-side and now measured separately. Old gate numbers are
  start-anchored history.**
- **Unseen-wording gate is now the selection instrument (2026-09-22,
  job 6781995): v11's bias-4 speaking was script recall — 0/10
  answered on new questions, chatter through the whole background
  tape. v3 keeps the most real behaviour: 2/10 on unseen wording at
  p50 280 ms (timing circuit intact, coverage is the failure) and
  clean t3+t5. MORE weight on the same 80 lessons is the least
  supported next step; data-side or capacity (Codex F5-2/F5-3) lead.**
- **Capacity test ANSWERED (2026-09-23, jobs 6792730/6814111/6814321,
  EXPERIMENTS.md "Capacity test"): NEITHER small lever learns 4
  conversations free-run. Rows-only (~41k params): word placement is
  saturated — 98.5% exact onsets, 97.1% on HELD-OUT conversations —
  but cannot learn silence (chatter 33–88% of sil ticks). Rows+adapter
  (~1M): learns silence TOO well — fully mute on 2/4 trained lessons,
  exact onsets 26%. Complementary failures at the same join point at
  TEACHER-FORCED vs FREE-RUN exposure (the model never sees its own
  chatter in training) as the common cause — capacity is exonerated
  for placement, the adapter is not a free upgrade. Also fixed:
  freerun's ±1s metric was vacuous (window contained its own gold
  tick), adapter-vs-CPU-feats device crash. NEITHER anticipated
  branch of Codex's completion criterion fired (adapter did not
  learn) — the rows-only ladder is not the thing to end; the thing to
  change is what training exposes the model to.**
- **Self-history probe 1 ANSWERED (2026-09-23, jobs 6816545/6820600,
  EXPERIMENTS.md "Self-history probe"): the exposure lever WORKS —
  first lever ever to move free-run silence to 58–96% (trained AND
  held-out) — but unweighted it swings to the MUTE pole (exact onsets
  1.5%, chatter ~0) with a new `<t+2s>` thief. Probe 2 (cap-sh2, job
  6820759) adds `--word-weight 3`: if speaking returns while silence
  holds, that config passes a bar NO model has ever passed (onsets
  ≥90% + chatter <10% on trained 4) and becomes the Phase-1 fix
  candidate to scale to the 80-lesson ladder.**
- **Where this goes after v11:** if the speak-up lever works, the Phase-1
  checkpoint target is "v10's quiet control + v2-level speaking + some
  timing"; if it cliffs too, the fallback is data-side (word-denser
  lessons) or accepting v3 and moving up the stack (M2 real voice, serving).

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
  **Layer-A Phase-1 status (2026-09-21): shards + trainer built; first
  gradient chain in flight.** M1 driver done (audio flows during
  generation, append-only KV; feed+decode p50 144 ms vs 80 ms budget —
  polish deferred to serving, off critical path). `layera/prepare.py`
  shards verified: ~1.04 audio tokens/tick (13 tok/s ear rate ✓).
  Training OOM chain (3 jobs, root-caused — do NOT repeat): batched_mm
  experts keep ~24 GiB (gate_up) + ~12 GiB (down) per-token expert
  gathers alive per layer → 59 GiB model + 35.7 GiB = the whole 95 GiB
  card; gradient checkpointing does NOT fix it (backward recompute holds
  all gathers). Fix: `experts_implementation="eager"` for TRAINING jobs
  (per-expert loop, ~6 MiB); keep batched_mm for inference jobs.
  Also: thinker `text_config.tie_word_embeddings=False` → embed and head
  are SEPARATE tensors; checkpoint saves both deltas
  (emb_delta_new_rows + head_delta_new_rows). Jobs: 6728087/6728254/6728411
  OOMed, 6728455 (eager) SUCCEEDED (32 steps, loss 10.4→~0.9).
  **Probe v1 (held-out, job 6742556): `<sil>` 0% → 95–98% acc on unseen
  lessons; word pieces improved (not a silence collapse); `<t+Ns>` 0% —
  data gap (those lessons contained almost no timing tokens).**
  **v2 (job 6742755, 2026-09-21): 80 class-balanced shards
  (`prepare.slurm PER_CLASS=10`), 400 steps ≈ 57 min, loss → 0.19.**
  Probe v2 (6745751, timed_task held-out): `<sil>` 99.4–99.7%, word
  pieces 20–48%, **`<t+Ns>` STILL 0%** — root cause: ~30 timing positions
  in ~50k graded (0.06%). NOTE: v2 trained into `runs/p1-v1` (default OUT
  before parametrization) and was renamed; the true v1 32-step checkpoint
  survives in `runs/p1-v1-32step` (verified via the `args` dict saved
  inside each checkpoint).
  **v3 (job 6746779, timing×50 weighted CE): MIXED — `<t+30s>` 0%→50%
  acc (prob ~0.55) on held-out, FIRST timing learning; but cost: `<sil>`
  99.5%→53.8/67.7% on 2 of 4 held-out scenarios, words roughly halved.
  The 50× push drags the shared context; everything else pays.**
  Dose-response COMPLETE (all probes corrected, SHARDS_DIR set,
  classes=timed_task,barge_in, 8 held-out each):
  - v2 (1×/1×): t+30s 0% prob 0 | sil 99.5% | yield 0%
  - v4 (k1×w10): t+30s 0% prob 0.15 | sil 97-100% ✓ | yield 0%
  - v3 (k1×w50): t+30s 50% ✓ | sil 53-98% ❌ | yield untested
  - v5 (k5×w1): t+30s 0% prob 0.000 | sil 97-100% ✓ | yield PARTIAL
    (1/6 scenarios 100%, others prob 0.03-0.76 — first yield learning)
  **Conclusion: only the DATA×WEIGHT product matters (~50 effective =
  timing learns); concentrating it in weight (v3) breaks <sil>; yield
  responds to data-spread. <yield> was a second rarity token (≈0.02% of
  graded) no weighting covered until now.**
  **v6 RESULT (CORRECTED 2026-09-22: the run changed THREE things vs
  v4, not one — shards-v6 visit counts barge 150×/timed 100×/others
  20–30×, yield weighting ON, and unshuffled sorted order changes the
  last class seen; "k5" never meant 5× distinct data): t+30s prob 0.000
  (printed; rounded to 3 dp) — the w10 timing signal (v4: 0.15) was
  gone. Sil perfect (97-100%), words best of any run (0.10-0.39),
  yield weaker than v5 (max 0.35). The original "DATA×WEIGHT law is
  dead" conclusion is CONFOUNDED — no clean single-variable
  attribution. Suspected AdamW scale-invariance weirdness stands as a
  hypothesis only.**
  **THIEF NAMED (6756308, v3 rerun with failure-argmax histogram): the
  sil "damage" was never broken silence — it is <t+30s> OVER-FIRING at
  sil positions (029: 584/588 sil failures = t+30s steals; 014/017/018
  same pattern). Timing recall fine; precision broken. Per-position
  weights boost the up-push but the down-push at sil positions stays
  weight 1.**
  **v7 VERDICT (6756309→6756310, k1 w30): t+5s LEARNED (1/6 scenarios
  100% acc prob 0.772; rest 0.047–0.372) with MILD over-firing (4–5
  sil steals/lesson — ~100× better than v3); t+30s NOT learned (prob
  0.014–0.032 — LOWER than v4's 0.15: at this dose the frequent t+5s
  grabs capacity first); sil 95.0–99.8%; yield 0%. Thresholds: t+5s
  crosses between w10 and w30; t+30s needs ~w50.**
  **v8 VERDICT (6756384→6756385, k1 class-cost 50): FAILED — label-blind
  loss. ∂L/∂z_k = p_k(c_k − E[c]); the −c_y term is a CONSTANT so the
  correct answer never enters the gradient — the only learnable content
  was "suppress rare tokens everywhere". Exam: t+*s prob 0.000 in all 8
  scenarios; sil acc 99.3–100% but prob flat 0.250 everywhere (learned
  constant, zero context); yield prob 0.250 acc 0; words ~0.
  --class-cost kept for the record; superseded by --thief-penalty.**
  **v9 LAUNCHED (2026-09-21): k1 w50 + --thief-penalty 10 — v3's proven
  up-push (label-dependent weighted CE) + a separate down-push taxing
  rare-token mass ONLY where the label is not rare (the thief's turf).
  Single-variable change from v3. FULL ABLATION LEDGER: EXPERIMENTS.md
  (per-run recipes, exam numbers, verdicts — keep current with every
  run).**
  **LIVE GATE #1 (v3, 6762718): FAIL 1/5 — constant tone, zero decodable
  words, never sil/yield/timing. FREE-RUN DIAGNOSTIC (6762979, v3 ckpt,
  training-format replay): free-run is CLEAN (sil 0.92–0.94 self-fed vs
  0.38–0.40 gold-forced; zero chatter; 81–110 recoveries to sil) → the
  live babble was the HARNESS FORMAT (chat envelope + processor
  <|audio|> path), not the model. Also: forced gold t+30s history PRIMES
  the thief (675 sil steals); free-run scores 0/6 at gold t+30s
  positions BUT fires t+30s 159× at wrong sil positions (61+98,
  corrected 2026-09-22 — the "needs the primer" story is not
  supported). FIX: duplex.py v2 + LayerATarget v2 — live now
  mirrors prepare.encode_audio (no envelope, 4 s-window ear features
  into <|audio_pad|> slots, one beat per feature, timestamped output
  tape). Live-gate rerun on the same v3 ckpt = the A/B. Full tables:
  EXPERIMENTS.md.**
  **LIVE GATE #2 (v3, format-fixed + per-test reset, 6763486): PASS 2/5
  — t3 ✓, t5 ✓ (0.72 s vs 22.0 s pre-fix); t1 answers but p50 14.9 s
  late, t2 silent after barge, t4 answers during part 1. Two harness
  bugs fixed en route: cross-test session leak (KV/clock carried
  between tests — T3's "babble" was stale-conversation continuation;
  Target.reset() per test collapsed it 562→3 word beats) and render
  tape overflow. Failures are now MODEL quality (sparse/early
  speech), not plumbing. Timing over-firing: 79/9/8 timing beats per
  test, silent — no metric cost yet.**
  **v9 VERDICT (6758553→6758554, k1 w50 + thief-penalty 10): FAILED —
  over-suppression ("the Monk"): sil 99.9–100% acc / prob 0.958–0.995
  (best ever) but t+30s, t+5s, yield, words ALL ≈0.000. The tax fires at
  every non-rare position (~1500/lesson) vs the up-push's ~1/lesson, so
  pen 10 swamped w50. Dial bracketed: pen 0 → timing learns + ~150
  steals/lesson; pen 10 → nothing. Next probe: pen 2. Ledger +
  post-mortems: EXPERIMENTS.md.**
  **v10 VERDICT (6762715→6762716, k1 w50 + thief-penalty 2): rare-token
  thief FIXED (1–2 steals/lesson, bar ≲5) and sil is the best of any
  w50 run (99.2–100% acc, prob 0.93–0.98) — but the tax's cliff sits
  between pen 0 and pen 2: t+30s died anyway (prob 0.003–0.013, 0/6),
  t+5s weak (prob 0.010–0.352, 0/4), yield 0. THE BOTTLENECK MOVED: the
  dominant exam error is now <sil> eating word positions (21–30/lesson)
  and words are below even unweighted v2 (0–19.4% vs v2's 20–48%) —
  every weighted run degraded speaking. The gate's failing tests (t1/t2/
  t4) are speaking failures; passing (t3/t5) are quiet tests. Live gate
  on v10 launched for the A/B (6763949). DECISION POINT: next GPU-hours
  go to word-beat reliability, not more rare-token/countdown surgery.**
  **LIVE GATE #3 (v10, 6763949): PASS 2/5 but MORE muted than v3 — 0
  word beats in all five tests (v3: ~9/test), 1 timing beat total; t5
  false engagement improved to 0.0 s, but t1 has no answers at all.
  THE EXAM PREDICTS THE GATE: word-slot sil-steals on the 10-min exam
  mapped exactly to live muting — the exam is now a trusted gate proxy.
  v2 (unweighted best-speaker, no timing) live gate launched (6764321)
  — if it beats 2/5 the T-gate needs only word+sil decisions and the
  recipe question becomes "inject timing without damaging speaking".**
  **LIVE GATE #4 (v2 baseline, 6764321): PASS 2/5 — ALL THREE ckpts
  pass 2/5; the gate is bottlenecked on speech production, not style.
  v2 is the only one that ever replied post-barge (1600 ms vs 800 bar;
  barge→sil 1760 ms) but it ALSO mutes on t1 (0 beats) and misplaces
  speech on t5 (10 word beats into background chatter, 0.48 s). So
  under-speaking is STRUCTURAL: sil covers ~95%+ of training ticks and
  every model, even unweighted, learns a silence attractor. Full
  three-way table: EXPERIMENTS.md. Next lever: weighted-CE on WORD
  positions (weight 1 today vs 95% sil background), doses bracketed.
  AWAITING USER DECISION on the next training run (stop-agreement).**
  **USER DECISION (2026-09-21): speak-up dose approved. v11 LAUNCHED
  (6765121 → exam 6765122, chained afterok): w50 + thief-pen 2 +
  --word-weight 3. Bracket plan: words flat on exam → 10; sil breaks →
  1.5. Exam is the decision instrument; live gate rerun only if the
  exam moves. New dial lives in train_p1.py (--word-weight) +
  train-p1.slurm (WORD_WEIGHT env).**
  **EVAL-GATE PATH BUILT (2026-09-21, pre-verdict): `layera/duplex.py`
  (DuplexThinker — M1 append-only tick loop + checkpoint deltas +
  trained-model emission decisions; mask = chat envelope only, the
  duplex ids MUST stay decodable since register_into marks them special)
  + `evals/harness/targets.py` LayerATarget (1 s chunk → 12-13 ticks;
  word beats emit a synthetic 220 Hz tone — voice is M2 — so the energy
  metrics measure speaking DECISIONS; n_timing/n_yield logged per
  chunk) + `evals/layera-live.slurm` (CKPT env, TESTS env). Run:
  `sbatch evals/layera-live.slurm` after a cluster `git pull`. Replay
  metrics score T1–T5 from the wav envelope exactly as for MiniCPM.**
  (v7/v8 done — verdicts above; per-run ledger lives in EXPERIMENTS.md.)
  Probes use
  env inheritance (NOT `--export`) for comma values — `--export` splits
  on commas, which silently truncated CLASSES=timed_task,barge_in to
  timed_task in probe 6746780. **probe-p1.slurm GOTCHA: always pass
  SHARDS_DIR** — without it the probe silently includes TRAINING rows
  (header prints "training set was 0 shards"); first v4/v5 probe attempt
  (6749312/6749314) was discarded for exactly this. Corrected:
  6751164 (v4) / 6751180 (v5). Checkpoints kept:
  runs/p1-v1-32step, runs/p1-v2-400step-unweighted, runs/p1-v3, p1-v4, p1-v5.
  **Hours-scale begun (2026-09-18): `synth-big1` = 256 scenarios, 2.2 h,
  1h24m GPU (0.85 GPU-h/audio-h), 182/256 pass (71%), WER median 0.018 /
  p90 0.111.** **VOLUME TARGET CLOSED (2026-09-19): 50.43 h gate-passed /
  6,388 scenarios across big1–31** (chatterbox 53%, minicpm 47%; all 42
  voices; class coverage 274–942 per class — lighter classes flagged for
  Phase-1 sampler weighting). Factory now runs on-demand for Phase-1;
  quality tail (class rebalance, optional Sesame engine) as needed.
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

- **ADR-0003 (2026-09-20, supersedes ADR-0002's bright line):** provenance
  classes replace the closed-API training ban — open / closed-labeled /
  closed-content (owner sign-off). Jev may label training data; streams
  tagged at creation; purge-by-provenance retained. TypeSafe Jev wired:
  `evals/harness/jev_scorer.py` (use case 1 live — semantic verdicts
  alongside energy metrics), SDK overlay `pylibs-typesafe`, key at
  `$DL/.typesafe_key`. Use cases 2-4 (bank audit, sampler scores, Layer-2
  labeling) queued.
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

- **Communication (user rule, 2026-09-20): plain language always.** Explain
  any technical term or jargon the first time it appears in a reply
  (barge-in = user interrupting; WER = word error rate, how different the
  transcription is from the script; overlay = a folder of extra Python
  packages we prepend so cluster jobs find them; KV cache = the model's
  running memory of the conversation). Keep tables/numbers, label them in
  everyday words. Applies to ALL agents working in this repo.
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
