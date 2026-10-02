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
- **Self-history probe series (2026-09-23, EXPERIMENTS.md "Self-history
  probe"): the exposure lever WORKS but has a split personality.**
  Probe 1 (cap-sh, self-history 0.5, unweighted): first lever ever to
  move free-run silence to 58–96% (trained AND held-out) — but it
  swings to the MUTE pole (exact word onsets 1.5%) with a new `<t+2s>`
  thief. Probe 2 (cap-sh2, + word-weight 3, job 6820759): **speech
  PARTIALLY returns on the barge lessons (13/21 and 10/20 exact,
  word acc 0.50–0.62, silence holding ~57% — the first config where
  speaking and stopping coexist), but the t+30s/t+10s thief eats the
  timed_task lessons (silence 34%, up to 505 steals; total exact
  onsets 25/68 = 37% vs the 90% bar). FAIL, but the shape is right.**
  Probe 3 (cap-sh3, job 6827932) added `--thief-penalty 2` (v11's
  recipe + self-history): **FAILED WORSE — the tax re-muted the model
  (4/68 = 5.9% exact, 2/4 lessons fully silent) while the thief
  SURVIVED (42 t+30s steals) — the same cliff the ladder hit at v10:
  penalty buys quiet, not precision.** Probe series verdict: the
  exposure lever works and word-weight 3 revives speech, but the
  config is knife-edge dose-response, not a stable fix.
- **OWNER DECISION (2026-09-24): the next lever is the DATA side —
  word-dense lessons attacking the ~95%-silence root.** Built and
  committed: `dense_dialog` (rapid Q/A, 71–72% assistant speech, new
  hand-written 20-pair bank) and `dense_barge` (interruption teacher
  in the dense regime, 90%+ speech), both through the existing
  render→filter pipeline (`CLASSES=` env in synth-render.slurm).
  Batch `synth-dense1` (per-class 8, seed 907, chatterbox) rendered
  2026-09-24. Probe 5 (probe-2 recipe verbatim on the 4-dense-lesson
  rig, job 6849442) COMPLETED and scored 2026-09-25: **FAIL on onsets
  (17/304 = 5.6% exact vs 90% bar), PASS on chatter (7.5%, just under
  the 10% bar).** Not muted — that's new — but placement is
  uncorrelated with gold: the ±1 s hit rate (52%) is at/below what its
  own chatter would produce by chance. Right amount, wrong places.
  NEXT: owner decision — word-weight 6 probe on the dense rig
  (pre-registered next dose, ~5 h GPU), or treat the labeling contract
  (one word tick per spoken word) as the suspect and discuss the spec
  change. Audition audio for dense-sh1 → `runs/model-audio-dense-sh1`.
- **PROBE 6 (dense-sh2 = ww6, job 6865063, 2026-09-26): FAIL both
  bars — dose went the wrong way.** Exact onsets 4.6% (worse than
  ww3's 5.6%), chatter 10.6% (over bar), and the <t+5s> thief
  exploded onto silence (993 steals; silence acc 0.91 → 0.22–0.44;
  one lesson bistably went near-mute instead). Loss-side dosing is
  REFUTED; per the pre-registered tree the labeling contract (one word
  tick per spoken word, ~85% of grid ticks are silence) is now the
  prime suspect. NEXT: owner decision — per-tick speech re-labeling
  (SPEC change, rebuild shards + one probe ~5 h) vs move up the stack
  (M2) accepting Layer-A timing as-is.
- **OWNER DECISION (2026-09-26): v2 label contract — per-tick speech.**
  New `<speak>` token (tokenizer/vocab.py): every tick inside an
  assistant utterance is WORD or `<speak>`; between-utterance gaps keep
  true `<sil>`; only SIL upgrades so timing tokens/yields keep their
  slots. Realized density: speech labels 59.7% vs sil 39.9% (was
  12%/88%). word-weight back to 1.0 (it only compensated v1 sparsity).
  freerun_diag v2 metrics: run-onsets + speech P/R/F1 (unit tests in
  evals/harness/test_freerun_metrics.py, all green). Old shards (v1
  labels) stay valid for v1 comparisons; `shards-dense2` = v2 rebuild.
  NEXT: build shards-dense2 → probe 7 (dense-pt1): SH 0.5/ww 1.0 on
  the same dense rig, pass bar ≥90% exact run-onsets + <10% chatter →
  audition + matched-voice render as usual.
- **PROBE 7 VERDICT (dense-pt1 = v2 + ww 1.0, job 6885910): FAIL —
  re-muted on balanced labels.** 3 of 4 trained lessons emit `<sil>`
  on every speech tick (speak acc 0.000, speech F1 0.000);
  dense_barge-004 babbles `<speak>` (acc 0.849 but false speech on
  166 quiet ticks, word acc 0.026). Held-out F1 0.00–0.16. Dose law
  now clean across BOTH contracts: ww1 → mute (probes 1, 7), ww3 →
  speaks (probes 2, 5), ww6 → babble+thief (probe 6). Label balance
  is NOT the mute attractor — loss shape is (sil/speak = one cheap
  token; words = the whole BPE vocab). Input-evidence insight: the
  model never hears itself, so its own history is the only self-speech
  evidence — under SH a collapsed model feeds itself silence.
  Step-8 recollect: untrained model picks words everywhere (450/450);
  by end it learned sil mastery and let go of speech. NEXT: owner
  decision — probe 8 = v2 + ww 3 (the only recipe that ever spoke,
  single variable vs probe 7) vs SH 0 (always-gold history while
  learning) vs move up the stack.
- **OWNER STRATEGY (2026-09-27): bounded ladder + M2 in parallel;
  the allocation must NOT lapse.** Allocation ends 2026-10-21. Goal:
  by ~2026-10-14, a model that speaks and takes turns audibly
  correctly through the M2 live serving loop, with remaining node-hours
  spent on the full-data run that gets there. Layer A is worth
  perfecting ONLY insofar as it fixes the live duplex experience by
  ear (owner's condition); if the ladder can't get there, pivot the
  hours instead of grinding probes. Probe 8 (dense-pt2 = v2 + ww 3,
  job 6901314) is the MUTING ENDPOINT: speaks → ladder moves to
  placement; mutes → ww exhausted, next is SH 0 (teacher forcing);
  if SH 0 also fails → redesign, not more probes. Probe 7's audition
  render is on the gallery (runs/model-audio-dense-pt1, matched
  voices; mostly-quiet samples as expected, barge-004 babbles).
- **PROBE 8 VERDICT (dense-pt2 = v2 + ww3, job 6901314): FAIL pass
  bar — but the MUTING QUESTION IS ANSWERED: ww3 revives speech
  (speak acc 0.000→0.30 on dialog-001); the model cannot PLACE it
  (word acc 0.02–0.05 ~random, exact onsets 1–3 of 3–5, chatter over
  bar). Failure moved from "won't speak" to "speaks at wrong times" →
  per the owner endpoint rule the ladder SHIFTS TO PLACEMENT. Next:
  probe 9 (dense-pt3) = v2 + ww3 + SH 0 (single variable vs probe 8):
  teacher forcing gives complete self-speech context while placement
  forms; exposure later. M2 milestone 1 landed in parallel: real-time
  tick parity PASS (clock changes nothing); kernel bench: batched_mm
  stays (eager 3× slower AND 58% argmax flips); piece-loop arithmetic
  says a SPEAKING model at ~57 ms/forward cannot sustain the 80 ms
  budget during words — decode speed is the M2 problem the moment
  Layer A delivers a speaker.
- **PROBE 10 VERDICT (dense-pt4 = pt3 rows + SH 0.5, job 6906391):
  CURRICULUM FAILED — placement decayed AND chatter remained.** Exact
  onsets 5/5 5/5 4/4 3/3 -> 0/5 3/5 1/4 1/3 (trained), two lessons
  fully mute; recollection log shows the in-training model flip-
  flopping chatter <-> mute across self-exposure recollections
  (1285/1344 speech-first -> 2/1344 -> 390/450 -> 2/304), so training
  saw contradictory self-histories and lost the placement it started
  with. Per the endpoint rule: REDESIGN, not more probes. Redesign
  target = the exposure distribution: gold-schedule self-history
  (gold words at gold ticks in history, gold silences as targets)
  instead of recollected free-run. Decision brought to owner
  2026-09-27. Audition renderer fixed for chatter era (soft limiter +
  solo model.wav per scenario; dense-pt3 re-render = job 6908524).
- **PROBE 11 VERDICT (dense-pt5 = pt3 rows + corrupt-gold SH 0.5, job
  6911469): FAILED the bar, but the trade-off is now LAW.** Corruption
  taught recovery (sil acc 0.5-0.9 vs probe 9's ~0-0.36) yet exact
  placement decayed again (2/5 3/5 1/4 1/3) and chatter stayed 65-94%
  of quiet ticks. Three-point law: the placement mapping lives exactly
  in clean-gold history; ANY training-side history corruption erodes
  it ~1:1 against discipline. Bottleneck looks structural (one head,
  one delta, ww3 word pole) -> decision brought to owner 2026-09-28
  (aux activity gate vs speaking-state channel vs stopgap+scale).
  Stopgap audition LANDED: pt3 + min-gap 4 + user-energy gate (hold
  3) = all onsets within +-1s, chatter cut 3-10x, words 297-887 ->
  60-188; gallery: runs/model-audio-dense-pt3-gated. TTS overlay dir
  is pylibs-minicpm (no "o") — two jobs died on the wrong name.
- **GATE 1 VERDICT (aux activity gate on FROZEN dense-pt3, job
  6912904): a real learned muting dial, but its scores don't rank gold
  onsets above chatter — bar still FAILED.** Gate trains cleanly
  (acc 0.990; first try 6912797 OOMed: batched_mm gathers ~43 GiB at
  training seq lengths — eager experts, same as train_p1). Sweep
  (thr 0.35/0.5/0.65 trained; 0.5 held): dialog-001/dialog-002 FULLY
  MUTED at every threshold (gold onsets score <0.35 — gate P printed
  per-onset now), other lessons chatter 8.8-44% (barge-001@0.65 passes
  the chatter half alone at 8.8%) but exact onsets 0-2 per lesson
  nowhere near >=90%. Four-point law (9/10/11/gate1): the trap follows
  the training-STATE distribution, not the parameter grouping — the
  gate only saw gold + bounded synthetic-corrupt states, decode visits
  pt3's own compounded states (all-sil history at first onset,
  always labeled 0 in training, true label 1). Audition render
  submitted (dl-dgaud): gate-0.35-only + gate+stopgap-rules combos,
  full TRAIN4, -> runs/model-audio-dense-pt3-gate035 / -gate-combo.
  Candidate next (owner decision): DAgger-style gate round 2 — train
  on pt3's own free-run rollout states with gold labels (~40 min).
- **Listenable checkpoints (owner request, 2026-09-23): audition
  renders of the four capacity/self-history checkpoints free-running
  the 4 trained conversations — the model's words spoken by TTS at the
  exact ticks it chose, over the conversation audio — into
  `runs/model-audio-*` (gallery layout). First job (6827931) died on a
  MiniCPM-o TTS internal crash mid-word; renderer now skips bad words
  (retry once, then continue). rsync to the Mac, then
  `python -m viz.listen` → http://localhost:8765. The gallery now
  binds BOTH loopbacks (dual-stack socket): some browsers resolve
  localhost to ::1 first and showed ERR_CONNECTION_REFUSED against the
  old IPv4-only bind (fixed 2026-09-24).**
- **Where this goes after v11:** if the speak-up lever works, the Phase-1
  checkpoint target is "v10's quiet control + v2-level speaking + some
  timing"; if it cliffs too, the fallback is data-side (word-denser
  lessons) or accepting v3 and moving up the stack (M2 real voice, serving).

## Current status

- **Runbook 00/01: DONE (Issue #1 closed).** Agentic SSH via
  `ssh u6si.aip2.isambard` (Clifton cert window is 12 h and does NOT
  auto-renew on attempts — corrected 2026-09-26 after an expired cert
  blocked all ssh for 5 h while jobs ran fine on the nodes. On
  `Permission denied (publickey)`: check
  `ssh-keygen -L -f ~/Library/Caches/clifton/u6si.aip2.isambard-cert.pub | grep Valid`;
  renewal needs the owner to run `clifton auth` and approve the Keycloak
  device URL in a browser within the code's few-minute lifetime). ARM
  env verified on GH200; Qwen3-Omni thinker smoke passed. Env
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
- **GATE 2 (DAgger) VERDICT (job 6916784): diagnosis confirmed — mute
  fixed, onsets 16/17 exact at thr 0.5 (bar-passing) — but chatter
  regressed to the other wall (31-77% trained, 52-82% held; bar <10%),
  threshold-flat on dialog lessons. Bar still FAILED.** Round 2
  free-ran TRAIN4 with gate1 active, buffered 2400 on-policy proposal
  states with gold labels, retrained; gate P at gold onsets jumped to
  0.66-1.00 (gate1: <0.35). Gate1+stopgap-rules audition (6916786):
  FULL MUTE, 0/21 exact — rules compound gate1's over-blocking;
  gallery batches model-audio-dense-pt3-gate035 / -gate-combo.
  Composite probe 6922786 (gate2@0.5 + min-gap 8/16 + user-gate
  hold 3) measures the last decode-side lever: gap divides chatter
  ~1:gap, gold run-gaps >=93 ticks so responses can't be clipped.
  Base model proposes words on ~91% of quiet ticks — the speech-rate
  prior itself is the residual problem (gate round 3 w/ iterated
  DAgger is the training-side candidate).
- **GATE2 COMPOSITE VERDICT (job 6922786): decode dial EXHAUSTED.**
  gate2@0.5 + min-gap 8/16 + user-gate fixed chatter as projected
  (held lessons all 5.2-9.7%, under bar) but strangled responses:
  min-gap counts from EVERY accepted word and gold words arrive every
  ~4-5 ticks inside a response, so follow-up words are blocked, the
  fed-back sil diverges the run, exact onsets 2/17 trained / 0/15
  held, recall 0.93 -> 0.19. Four configs now mapped: gate1=mute,
  gate2=chatter, gate2+rules=strangled, rules-alone=compromise. You
  cannot rule your way out of a wrong prior. What's left: training
  (gate round 3, iterated DAgger w/ quiet rebalance) or acceptance
  (M2 on gate2@0.5 + user-gate). Decision brought to owner.
- **M2.2a RESULT (job 6923669): parity property HOLDS on the serving
  decode — paced vs dry picks identical, 0 mismatches on all 4
  TRAIN4 lessons with gate2@0.5 + user-gate in the loop; picks-m2-
  serving saved as the live-loop reference. The speed gate FAILED,
  now precisely measured: quiet ticks p50 55.3 ms (fits 80 ms);
  SPEAKING ticks p50 417 ms, p95 ~430 ms, max ~497 ms (piece loop,
  ~3.3 pieces/tick avg) -> adherence 3.8-10%.** Milestone 2b
  (decisions on time, pieces in arrears) is the validated fix;
  eager/compile kernel bench deferred into the 2b job (job exited
  at arm 1 under set -e; the loop shape changes in 2b anyway).
  Gate round 3 first submission timed out at 2h (12-lesson rollout
  ~12 min/lesson; 9/12 collected, no checkpoint) -> walltime 4h,
  resubmitted.
- **GATE 3 VERDICT (12-shard DAgger, job 6952939, 10,119 states):
  FIVE-POINT LAW — three gate rounds are three operating points on
  ONE frontier (gate1=mute pole, gate2=chatter pole w/ perfect
  placement, gate3=middle: held chatter 21-28% but onset P collapsed
  to 0.01-0.35, exact 0-2/15). Buffer acc 0.869 (not memorizing) and
  still fails -> the trunk state alone cannot separate turn-start
  from chatter in this recipe family. Murmur fix = state coverage =
  the scaling run. Demo config: gate2@0.5 + user-gate. Optional
  round-4 hypothesis: gate input += user-audio energy.**

## 2026-09-30 — scaling run prepared (owner-approved "scaling run + M2")

One command launches the whole chain, cert-lapse-proof (each stage
self-chains the next with sbatch --dependency=afterok):
`sbatch evals/scale1-data.slurm`
1. **scale1-data** (12h): banks (Qwen3-Omni text-only, ADR-clean;
   reused if present) -> 240 dense timelines (120/class, seed 23 —
   dense_dialog+dense_barge ONLY, the classes every probe measured)
   -> MiniCPM render, 2 workers. Timeout-safe: clip cache resumes;
   then chains filter -> shard jobs.
2. **scale1-shard** (12h): prepare over filter-passed lessons;
   resume-safe (skips existing shard .pt files — new). Chains train.
3. **scale1-train** (24h): probe-9 recipe UNCHANGED (v2 labels,
   ww3, SH0) on ~224 lessons, 2000 steps (~9 passes/lesson vs
   probe9's ~100 — same recipe, coverage instead of memorization),
   --save-every 200 (new; tmp+rename atomic). Held-out = idx
   113-120/class, closing free-run on 4 of them. Calibrate STEPS
   vs probe9 elapsed (sacct 6905339) before first submit; resume
   = TRAIN_INIT=<ckpt>.
Trainer changes: train_p1 --save-every; prepare skip-existing.
BEFORE SUBMIT: `! clifton auth` (cert lapsed again), git pull on
cluster, md5sum check, then check walltime caps + probe9 elapsed.

### Research passes (2 background agents, 2026-09-30) — what prior work says

**Serving side (M2):** Moshi's Mimi codec = 80ms causal frames,
zero lookahead, 200ms end-to-end — our 80ms tick is a published,
working design point. Rolling-window incremental encode = standard
cached-left-context streaming; the hard 4s drop-oldest truncation
is OUR untested part (M2.3). Piece bank = "limited-domain
concatenative/unit-selection synthesis", a named 30-year-old
technique; known join artifacts (pitch/phase/energy mismatch) with
cheap mitigations: cut units at pitch epochs + short crossfade.
Barge-in in deployed stacks = cut at output-buffer granularity
(~20-100ms), hard cut with a few-ms gain ramp; no one fades.
**M2 actionables: 2c piece bank should cut at pitch epochs and
crossfade joins; barge-in per-piece check matches practice.**

**Training side (scaling run):** field floor for real duplex
behavior is ~2,000 REAL hours (Fisher; Moshi: 7M hours mono did
NOT buy placement — it needed Fisher + RL). Largest synthetic
push: SyncLLM 212k synthetic hours (helped meaningfulness; timing
still collapsed out-of-domain 0.43 -> 0.23). Our 240 lessons ≈
single-digit AUDIO-hours — tiny in hours, but (a) our trunk is
already fully speech+language-capable (we train ONLY 10 timing
tokens), (b) our lessons are saturated with turn-taking events
(Fisher hours are sparse in them), (c) direction is precedented
(synthetic duplex-format data at scale helps), magnitude untested.
Aux turn-taking heads on frozen trunk states ARE published
(Freeze-Omni: linear 3-state head on hidden states — near-identical
to our gate); our exact failure mode (can't split turn-start from
chatter on those states) is NOT published; nearest analog dGSLM
answered nearly every pause (TOR 0.93). Kyutai's own fix for
timing was RL with timing rewards (interrupt latency 1.38 -> 0.46s)
— the known complement if scaling alone lands short. Behavior-
balanced mixtures are the other lever (our scenario classes ARE
that taxonomy; currently only the 2 dense classes scale).

## 2026-09-30 (later) — SCALE1 CHAIN LAUNCHED + RL plan + ADR amended

**Chain is RUNNING.** Data job 6960233; watcher armed (stage
transitions + terminal states; chain is cert-lapse-proof internally).
Calibration measured before launch: probe9 = 400 steps / 1h05m35s
(~10 s/step) -> 2000 steps ≈ 5.5 h (24 h budget fine); shard
encoder = 12 lessons / 67 s (~5 s/lesson -> ~20 min for 240);
prior render 4.5 min warm; clip cache holds 2,753 clips; workq
MaxTime=UNLIMITED. md5 verified all 7 touched files local==cluster.

**ADR-0003 Amendment A (2026-09-30): closed-content cleared
generally** — enterprise agreement licenses closed-API content for
training; provenance tags stay as hygiene, not a gate. Agent rule:
stop citing public provider terms here (owner: they are not the
current authority). Wave-2 option now open: content banks from the
ChatGPT API (same banks.json format, banks-gpt module = small) —
decide at wave 2; the running chain uses Qwen3-Omni banks.

### RL exploration (owner 2026-09-30: "explore the right policy and
### training method while we have compute")

Two-probe ladder, cheapest first, START AFTER scale1 reads out:
- **RL-probe 1 — head-DAgger (~hours, reuses gate-2/3 machinery).**
  train_gate's DAgger rollout already produces (on-policy state,
  label) pairs; relabel them GOLD for the WORD HEAD and train the
  head rows on that buffer (new small trainer, e.g.
  layera/train_head_dagger.py; save rollout states once to
  runs/dagger-states). Rationale: chatter = the ~91% quiet-word
  prior never corrected on visited states; DAgger gives exposure
  WITHOUT poisoned context (self-history probes failed on corrupt
  inputs; this keeps inputs real, labels gold — exactly what fixed
  placement ranking in gate round 2). Pass bar v2 unchanged: held
  chatter <10% AND exact onsets >=90%.
- **RL-probe 2 — segment GRPO with timing rewards (~1-2 GPU-days to
  first signal; Kyutai's published lever).** Mine short behavior-
  dense segments (60-120 ticks around pauses/interrupts) from
  timelines; K=8 group rollouts per segment; reward = bar-v2 metrics
  (exact onset +, chatter −, mute −); group-normalized advantage;
  update ONLY the 10 new-token rows (+gate). Kyutai precedent:
  interrupt latency 1.38 -> 0.46 s. Cost wall = full-lesson rollouts
  (12 min/lesson) — segments are the fix; batch the K samples.
- Decision rule: probe 1 first. Chatter fixed -> stop. Partial ->
  GRPO. Both fit the allocation before it ends 2026-10-21.

## 2026-10-01 — scale1 verdict (FAIL), chain postmortem, gate1-on-scale1 launched

- **Scale1 readout: the pass bar failed both halves** (5/17 exact
  onsets, false speech 41-75%, one fully-mute held lesson). 33x
  coverage on the unchanged recipe did not break the five-point
  frontier — same trade at scale, per-lesson pole mixing. Full table
  in EXPERIMENTS.md "SCALE1 (the scaling run)".
- **Three chain incidents, all fixed + pushed**: shard --limit 8 bug
  (commit 9cc4213, invalid 8-lesson train cancelled ~2.5 GPU-h in);
    rerender manifest-append duplication (scale1-rerender.slurm);
  dialogroll part-file naming (globbed). Dialog filter collapse is
  SYSTEMATIC (base-rate clip babble; re-rolls +5/round, exhausted).
  Render-side per-clip ASR check = the queued fix if dialog coverage
  matters. Gallery: runs/synth-scale1 synced (240 lessons audible).
- **LAUNCHED: evals/scale1-gate1.slurm** — gate round 1 on the scaled
  trunk (runs/scale1-pt1), UNGATED 8-lesson DAgger rollout (4 dialog +
  4 barge), sweep mirrors gate rounds 1-3, held = the same 4 lessons
  for direct comparison with the ungated verdict.
- **Next after gate1-on-scale1 reads out**: RL-probe 1 head-DAgger
  (train_gate rollout states -> gold labels for the WORD head rows;
  needs layera/train_head_dagger.py written), then RL-probe 2 segment
  GRPO per the plan above. M2 2b (piece-budget scheduling) still the
  parallel track toward the Oct-14 audible goal.

## 2026-10-01 (part 2) — gate1-on-scale1 FAIL, RL plan hardened, probe 1 running, mitigants in place

- **Gate1-on-scale1 verdict: FAIL** (commit 47a52c2). 0/17 held exact;
  gate P at gold onsets bimodal PER LESSON (0.98-1.00 vs 0.02-0.09) =
  lesson-level property. The five-point law EXTENDS to the 33x scaled
  states: the trunk state alone cannot separate turn-start from
  chatter, and a side gate cannot fix it. Full table in EXPERIMENTS.md.
- **RL plan reviewed against research BEFORE compute** (f274f9d,
  docs/notes-rl-timing-research.md): Kyutai's full-duplex RL paper
  (arXiv 2606.11167 — G=16 GRPO, reward-decoupled normalization,
  context curriculum, joint-axis batches, interrupt latency 1.38->0.46s)
  + DAPO failure modes (arXiv 2503.14476). 10 adoptions recorded.
  Critical assessment (owner challenge): Kyutai is a prior, not gospel
  — their RL SHARPENED existing behavior, ours must CREATE it (raw
  passes here range from 1,161 word-proposals/lesson to ZERO); GRPO's
  honest role = FINISHER on dialog (base rates non-zero), starved on
  barge until DAgger injects behavior; the judge-free inversion stands
  (scripted gold makes their LLM judge unnecessary).
- **RL-probe 1 (head-DAgger) RUNNING**: job 7001927 on scale1-pt1,
  8-lesson raw-decode rollout -> train ONLY the 11 duplex head rows
  (layera/train_head_dagger.py, commit 35426b3) -> UNGATED freerun on
  TRAIN4 + the same HELD4 as scale1 (direct comparison). Rollout
  polarization CONFIRMS per-lesson pole mixing live: dialog-001/010
  ~1,100 proposal states (chatter pole), barge-010 = 0 and
  dialog-012 = 2 (mute pole — DAgger gets NO states there: it cannot
  correct trajectories the model never leaves; known blindspot).
  Buffer ~3,600 states, chatter-side dominant. Readout ~2h from 13:20
  local.
- **Mitigants IN PLACE (7f45ef0)**, per the owner's risk review:
  - layera/panel.py — the regression GATE. Parses freerun logs
    (retroactive), reports exact/+-1s/chatter3/R3/full-mute per
    lesson, pairs candidate-vs-baseline; FAIL on exact/+-1s drop
    >5pt, chatter +50 ticks, or a NEW full-mute (baseline mutes are
    the bar's problem). Validated: reproduces gate1's 0/17 held
    verdict from the real log; self-gate passes. EVERY RL readout
    runs --gate-vs <scale1 baseline log>.
  - Fresh set RESERVED: barge-115/117/119 (never eval'd). GAP: no
    fresh DIALOG lessons exist (held dialog collapsed too; only
    116/118 sharded, both selection-burned) — fix rides the
    re-render decision below.
  - Reward arithmetic PINNED (notes #3): full +2.7 vs shortened +1.6
    vs mute -0.5 vs chatter <0; train_grpo.py MUST assert the
    dominations at startup or refuse to launch.
  - Leashes: KL logged every step (abort >10x initial), entropy
    floor (abort <20% initial), DAPO dynamic sampling for
    zero-advantage groups. Staged: pilot (hours, panel-gated) ->
    owner go/no-go -> 1-2 GPU-day run. Gallery audition via
    render_model_audio + viz.listen at every readout. Parity rule:
    any RL checkpoint in the demo config -> M2.2a re-run.
- **Pending owner decisions (bundled at probe-1 readout)**:
  (a) GRPO go/no-go on the panel-gated probe-1 numbers — if probe 1
  stalls, the decision is the LoRA escalation (notes #9), NOT more
  11-row spend; (b) dialog re-render with per-clip ASR check + NEW
  fresh dialog held lessons (fixes the 29/120 filter collapse AND the
  freshness gap; GRPO wants the segment pool either way).
- **Parallel always**: M2 2b (piece-budget scheduling) — the live
  loop is on the critical path to Oct-14 regardless of RL.

## 2026-10-02 (evening) — D1 + UGATE: the two decisive negatives; model-side levers CLOSED

Jobs 7019983 (D1 diag, 85s) + 7019984 (ugate direct-path, ~2h), both
COMPLETED after the PYTHONPATH=$DL/pylibs fix (7a2fd08; both new
scripts had dropped pylibs — transformers unresolved, died in ~10s).

**D1 (EXPERIMENTS.md "D1: STATE DIAGNOSIS")**: linear+MLP probes on
the 4,374 saved free-run states, ALL cells at chance (per-lesson,
pooled, AND leave-one-lesson-out; MLP=linear). Verdict: no turn-start
signal in the trunk states at all. Readout training dead in its
strongest form; retro-explains gate-MLP 1-3 + linear head-DAgger +
head-DAgger CE with one cause (the free-run speech schedule has
DECOUPLED from the script clock; the state encodes the model's own
pole, not the script's clock). **LoRA probe-1b CONTRA-INDICATED
(there is nothing for it to read out).** GRPO stays retired.

**UGATE (EXPERIMENTS.md "UGATE")**: scale1-pt1 + mic-energy VAD
user-gate on HELD4 — FAIL 3/4 gates at both holds. Chatter ~halves
(the channel is real) but onsets pay (116: 5/5->2/5 +-1s) and
dense-user barge lessons saturate the within-lesson percentile
threshold -> barge-113 NEW full mute (barge-114 was already mute).
TRAIN4 fine (dialog-001 4/5, barge-004 4/4; dialog-010 is its
pre-existing pole). Third confirmation of the same conclusion:
**the mute<->chatter frontier is a property of the TASK (script-clock
gold over never-per-clip-verified audio) + a model that cannot
represent the script clock — no mechanism on top moves it.**

**State of the ladder**: recipes (5-point law at 33x), readouts (D1:
chance), exogenous gating (ugate: same frontier) — ALL CLOSED with
data. Remaining levers are objective-level only:
- A. Label-quality audit (near-free): per-clip ASR vs script on the
  rendered assistant clips (clip granularity version of the filter's
  lesson-WER), then the pending dialog re-render + FRESH held
  dialogs. D1's decoupling is exactly what inconsistent gold predicts.
- B. M2 live loop = critical path to audible-by-Oct-14: causal VAD
  (absolute or running floor, NOT per-lesson percentiles — ugate
  showed why), 2b piece-budget scheduling (the 417ms wall),
  parity-gated promotion. The demo composite already ran end-to-end.
- 4% ear-clock skew (52 ear tokens vs 50 ticks per 4s) is a known
  wart of the right size (~150ms over 30s history) to smear onset
  information — fix belongs in the M2 live loop regardless.
- Audio of both ugate arms: job 7020187 -> runs/model-audio-ugate-
  {held,train}; rsync + viz.listen before believing the FAIL.
- Fresh-set rule unchanged: barge-115/117/119 NEVER eval'd before
  final readouts.
