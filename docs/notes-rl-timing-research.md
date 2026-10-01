# RL plan review — research notes (2026-10-01, pre-compute)

Two sources reviewed before the GRPO run; both change the plan.
Probe 1 (head-DAgger) is unaffected — it is supervised and already
running (job 7001927).

## Source 1: Kyutai, "Multi-Faceted Interactivity Alignment in
## Full-Duplex Speech" (arXiv 2606.11167, 2026-06)

The published instance of our exact problem: RL post-training of a
full-duplex speech model (Moshi, PersonaPlex) on the four interactivity
axes — pause handling, turn-taking, backchanneling, user interruption.
Their recipe:

- **Segments, not lessons**: per-axis extraction from Fisher +
  Seamless (structural criteria: utterance lengths, gaps, IPU chains);
  up to 2,000 segments/axis; a preceding context window (0->30 s,
  scheduled, sampled p=0.5) is prepended and MASKED from the loss.
- **GRPO**: G=16, group-normalized advantage, clip 0.2, KL beta=0.01
  to a frozen reference, LR 2e-7 cosine (FULL fine-tune, 32 H100s,
  FSDP), grad-norm clip 2, temp 0.7 text / 0.8 audio.
- **Loss on TEXT tokens only** (not audio tokens) — the text stream
  controls timing and content.
- **Reward-decoupled normalization**: each reward component
  standardized independently within the group, summed equal-weight —
  stops the dense timing reward from numerically dominating the sparse
  content reward.
- **Rewards are programmatic except one**: VAD/ASR-based delay and
  F1 terms + an LLM judge (Qwen3-235B) for semantic relevance, added
  only to turn-taking/interruption. Their judge exists because their
  segments come from REAL corpora (right content undefined). OURS come
  from scripts — gold content is known, so the judge's job is done by
  the grid. The rule-based reward here is complete, not simplified.
- **Results**: interrupt latency 1.377->0.461 s; pause AND turn axes
  improved JOINTLY (the frontier trade can break under consequence-
  level training); UTMOS speech quality unchanged; ablations: dropping
  the judge = largest degradation across metrics; dropping the KL =
  audio drift; single-axis training = the other axis degrades.
- **Caught regression**: PersonaPlex+Fisher became sycophantic
  ("Yeah"-compliance) — a data-mix side effect visible ONLY in
  cross-metric eval. Lesson: monitor metrics OUTSIDE the reward every
  eval; keep both our classes in every batch.

### Adoptions for scale1-GRPO

1. Reward-decoupled per-component normalization (adopt verbatim).
2. Context-window curriculum: prepend the lesson's prior ticks as
   frozen context, masked from the loss; grow 0->900 ticks over
   training.
3. Loss lands ONLY on duplex-token decisions automatically: frozen
   word rows make the importance ratio exactly 1.0 on continuation
   pieces (log 1 = 0 gradient) — uniform-trajectory advantage + frozen
   rows is self-consistent, no extra masking needed.
4. Onset reward as continuous delay shaping (theirs: -d, absence =
   worst) blended with our exact-tick bonus; components normalized
   separately so scales don't fight.
5. Segment mining mirrors their axis criteria translated to our grid:
   dialog class = turn-taking axis, barge class = interruption axis;
   BOTH classes in every batch (joint-axis training is what broke the
   pause-vs-turn trade — our five-point frontier's analog).
6. Degenerate-group handling upgraded from "skip" to DAPO's dynamic
   sampling (resample until the group has mixed outcomes).
7. LR regime for GRPO: 1e-6..1e-5 on the 11 rows (their 2e-7 was full-
   FT; ours is 11 rows but the GRPO gradient scale is unknown — start
   1e-5, watch KL), grad-norm clip 2, small KL.
8. Anti-shortening guard in the reward: word-coverage/recall term —
   ending responses early reduces chatter-tick exposure, so an
   unguarded reward can learn to under-speak (our equivalent of their
   semantic-collapse finding).
9. Escalation path validated: if 11-row GRPO stalls, LoRA @ tiny LR is
   the published middle ground before any full-FT thought.
10. Eval hygiene: add a FRESH held-out lesson set never used in any
    gate/RL selection decision (all gate rounds reused the same 8 —
    selection-overfit risk is real and now acknowledged).

## Source 2: DAPO (arXiv 2503.14476) + GRPO-tricks write-ups

Four failure modes of naive GRPO with verifiable rewards, and fixes:

- **Zero-advantage groups** (all K attempts score the same -> no
  gradient): dynamic sampling — discard and resample degenerate groups.
  (adopted, #6 above)
- **Entropy collapse** (policy diversity dies -> premature
  determinification, for us the mute-pole attractor): Clip-Higher —
  decoupled, higher clip for low-probability tokens. (adopted)
- **Sequence-length coupling**: token-level loss normalization by
  total batch tokens (DAPO) — matters less for us (fixed-length
  segments) but cheap to include.
- **Overlong truncation gaming**: n/a for fixed-length segments.

## Mitigations IN PLACE (2026-10-01, pre-compute) — the side-effect
## defenses, built before any GRPO launch

Principle: the reward cannot see the side effects, so the eval must —
and it must GATE, not just report.

1. **Regression gate — BUILT** (`layera/panel.py`, validated against
   the gate1 log: reproduces the 0/17 held verdict + flags exactly).
   Parses freerun_diag logs (retroactive: works on every past/future
   freerun without re-running), reports per-lesson exact / +-1s /
   v3-chatter / R3 / full-mute, and pairs candidate-vs-baseline
   occurrences. Gates: exact or +-1s fraction drops >5pt, chatter
   rises >50 ticks, or a NEW full-mute pole (baseline mutes are the
   bar's problem, not a candidate side effect; recovered mutes named).
   Exit code gates slurm steps. Every RL readout runs
   `--gate-vs scale1-baseline-log`.

2. **Fresh held-out designation** — barge-115, barge-117, barge-119
   (sharded, never used in ANY gate/scale/probe decision) are RESERVED;
   no eval touches them before final readouts. GAP, honestly recorded:
   NO fresh dialog lessons exist — of held-split dialog 113-120 only
   116/118 have shards (the tail_babble filter collapse hit held dialog
   too), and both are already selection-burned. Fix rides on the
   owner-decision re-render (per-clip ASR check + fresh dialog held
   lessons reserved from birth).

3. **Reward arithmetic pinned (shortening and mute must LOSE money)** —
   proposed rates: onset +1.0 exact / +0.3 within +-1s; chatter
   -0.02/false-quiet-tick; missed answer -0.5; word recall +2.0 x R.
   Segment-level checks (a ~15 s segment ~= 187 ticks):
     full honest answer  ~= +1.0 + 1.8 - ~0      ~= +2.7
     short answer (R~0.3) = +1.0 + 0.6            = +1.6  < full
     all-mute             = -0.5                  < any speaking policy
     chatter-everywhere   ~= +0.3 + 1.8 - 2.4..-3.7 < 0
   So completing lines strictly beats shortening (recall term's
   +1.4 marginal > ~0.2 in-window chatter exposure), speaking strictly
   beats muting, and timed speech strictly beats chatter. train_grpo.py
   MUST assert these inequalities at startup with the configured rates
   — if a rate edit breaks a domination, launch refuses.

4. **Training-time leashes (built into train_grpo.py at construction)**:
   log KL-to-reference every step; abort if KL > 10x its initial value
   (recipe mismatch = stop, not tune). Log policy entropy; entropy
   < 20% of initial = mute-attractor forming = stop. DAPO dynamic
   sampling resamples zero-advantage groups so mute-collapse cannot
   hide as "no gradient".

5. **Staged spend**: GRPO pilot first (small segment pool, a few
   hours, panel-gated) -> owner go/no-go -> only then the 1-2 GPU-day
   run. Kill switches above apply to the pilot too.

6. **Human gate**: fixed gallery per readout (TRAIN4 + HELD4; + FRESH
   at final) rendered via evals/render_model_audio.py and listened via
   `python3 -m viz.listen`. The ear catches grid-legal-but-awful
   behavior no token-stream scorer sees.

7. **Parity rule (standing)**: any RL checkpoint entering the demo
   config triggers an M2.2a parity re-run — noted now so it cannot
   bite at demo time.

8. **Critical assessment recorded** (owner challenge, 2026-10-01):
   Kyutai is a prior, not gospel — their RL SHARPENED existing behavior
   (ours must often CREATE it: barge-010 raw pass = 0 proposals,
   dialog-012 = 2); their numbers are calibrated to 32xH100 full-FT;
   their data realism (human timing statistics) does not transfer to
   our scripted gold; their eval breadth caught their sycophancy
   regression and we are only now building ours (#1 above). GRPO's
   honest role here: FINISHER, not creator — dialog base rates are
   non-zero (something to amplify: scale1 held dialog 5/17 exact,
   10/10 +-1s placement); barge base rate ~0 (DAgger must raise it
   first — probe 1 is exactly that). GRPO is justified as the only
   tool optimizing the trajectory-level bar directly (per-tick
   imitation compounds errors over 900 ticks and plateaus below the
   90% bar), NOT by exploration breadth (our gold is near-unique).

## What was checked and did NOT change the plan

- Q-network/critic: not needed at outcome-reward granularity (group
  baseline is the validated replacement; per-tick credit is the
  follow-up if outcome-level signal is too blunt).
- Learned reward model / LLM judge: correctly absent — our segments
  are scripted (gold content known); the judge in Kyutai's recipe
  covers exactly the part we get for free.
- Full-FT vs frozen-trunk: their full-FT worked at LR 2e-7; our
  11-row constraint is far more conservative and structurally protects
  placement and content; escalation ladder if needed is documented
  above.
