# Phase-1 ablation ledger

One row per training run: the recipe, the held-out exam result, the
verdict. **Keep this current with every run** — it is the memory of what
we tried so we never re-buy the same lesson. (Plain-language rule
applies: acc = how often the model's top pick was the right token; prob =
how much of its confidence it put on the right token, on average.)

**Writing convention (owner instruction, 2026-10-02):** every new
entry starts with a "Plain words:" paragraph — what is the problem,
what did we do, what did we find, what does it mean — readable on its
own; technical detail and numbers go below it. Every technical term
gets a plain gloss the first time it appears.

**What "the exam" is:** every run trains on 80 factory lessons, then is
quizzed on 8 lessons it NEVER saw (held-out). A run only counts if it
passes on unseen lessons — that is generalization, not memorization.

## THE SHORT VERSION (plain English, updated 2026-10-02)

**The problem:** the model cannot tell when it is its turn to speak.
On conversations it has never heard, it either butts in while the
user is still talking, or sits silent when it should answer — and it
almost never starts talking at exactly the right moment. Every fix so
far only trades one mistake for the other: make it quieter and it
misses its turns; let it talk and it interrupts. Nothing has fixed
the timing itself.

**What we proved on 2026-10-02 (two experiments, full entries below):**

1. **D1 — the timing sense is not inside the model.** We trained
   small pattern-detectors ("probes") to find one thought in the
   model's internal numbers: "it is my turn to speak now." A working
   detector scores near 1.0; a useless one scores 0.5 (coin flip).
   Every detector — simple or fancy, inside one lesson, across
   lessons, on lessons it never saw — scored coin flip. All the
   earlier "reader" failures happened for this one reason: **there
   was never anything there to read.** That family of fixes is closed
   for good (including the LoRA add-on idea).
2. **UGATE — even a fully external fix lands on the same trade.** A
   loudness meter ("is the user talking right now?") blocked the
   model from starting a word while the user speaks. Interruptions
   halved (the meter works) but the model then started its turns too
   late or missed them, and on user-heavy conversations the meter's
   cutoff — chosen from that conversation's own loudness levels —
   lands inside the user's speech, so the model got blocked almost
   always and went completely silent. A meter that knows only
   loudness cannot know whose turn it is.

**The conclusion (corrected 2026-10-02, after mapping the data
factory end to end):** the timing problem is in the TRAINING DATA
PIPELINE, and the precise gaps are now known:
- The filter DOES speech-to-text every clip (a Whisper model, per
  clip) — content *detection* exists. But a single bad clip rejects
  the WHOLE conversation, and nothing repairs it: the renderer speaks
  each clip exactly once (no retry at render time), and the only
  feedback loop was "purge the clip cache and re-roll whole
  conversations", which recovered just +5 dialogs per round before
  stalling at 29/120. The pipeline proceeds by DROPPING conversations
  — leaving smaller, selection-biased training sets — instead of
  fixing the clip that failed.
- Word timing inside a clip is ESTIMATED (characters → milliseconds,
  a documented v0 placeholder in the label builder), never audited
  against the clip's real word timestamps.
- The model's ear clock runs ~4% fast versus the 80 ms tick clock
  (52 ear tokens vs 50 ticks per 4 s), silently absorbed by
  truncation in the sampler — enough drift to smear timing over a
  30 s history.

**Was the previous data prepared well?** Partly — now measured
precisely:
- Content detection: strong (per-clip Whisper round-trip, per-channel
  word-error rates, babble gates; passing lessons show ~4% word
  error).
- Content repair: absent — one bad clip killed the conversation
  (this exact mechanism collapsed the dialogs to 29/120).
- Timing labels: derived from MEASURED clip durations (smarter than
  we thought), but word placement inside clips is an estimate and was
  never audited; plus the 4% ear-clock mismatch above.

**The solution (owner go-ahead 2026-10-02, both in parallel, under
the FAIL-CLOSED DATA RULE — see HANDOFF.md):**
1. **Fix the training data, and make bad data impossible to miss
   again.** Four concrete changes, mapped against the real pipeline:
   (a) *repair-in-render* — move the filter's per-clip check INSIDE
   the render loop: a clip that fails is re-spoken immediately (a few
   retries); a conversation is rejected only if a clip still fails —
   this converts the 29/120 dialog collapse into yield;
   (b) *word-timing audit* — Whisper word timestamps per clip vs the
   label builder's estimated placement; conversations whose word
   onsets drift beyond tolerance are re-labeled from the measured
   times or rejected;
   (c) *ear-clock fix* — stop silently truncating the 4% ear-token
   surplus; align the ear token rate to the tick grid;
   (d) *fail-closed asserts* — every training/evaluation job checks
   at startup that its lessons carry a passing quality record (clip
   verdicts + timing audit) and aborts otherwise. No more proceeding
   on unverified conversations — enforced in code, not promises.
2. **Build the live loop (M2) in parallel,** with the same discipline:
   its results only count on conversations with verified answer keys.
   Mic in, speech out, one 80 ms tick at a time — the path to
   something audible now, with the model we already have. The 2b
   scheduler (decisions on time, pieces in arrears) is implemented
   and its measurement job is queued (see HANDOFF 2026-10-02).

## Decision rule for the Phase-1 checkpoint

Pass the token exam AND the conversation gate:

1. **Token exam (probe):** timing tokens fire on unseen lessons (t+30s
   ideally; t+5s acceptable for the first gate) with `<sil>` ≥97% and
   over-firing bounded (thief steals ≲5 per lesson, vs v3's ~150).
2. **Conversation gate (T1–T5 live harness):** the checkpoint drives
   DuplexThinker through the scripted conversations; energy metrics must
   match the THRESH bar (reply after barge ≤800 ms, response p50 ≤1500 ms,
   no yielding on backchannels, ≤2.0 s false engagement).

## Dose-response series (all: 400 steps, frozen brain, 10 new-token rows only)

| run | jobs (train→exam) | recipe (data × weight) | t+30s | t+5s | `<sil>` | `<yield>` | words | verdict |
|-----|-------------------|------------------------|-------|------|---------|-----------|-------|---------|
| v1 | 6728455 → 6742556 | old data, 32 steps | 0% | — | 95–98% ✓ | 0% | ok | Smoke test: pipeline works end-to-end; timing data missing |
| v2 | 6742755 → 6745751 | 80 balanced shards ×1 | 0% prob 0.000 | — | 99.4–99.7% ✓ | 0% | 20–48% | Baseline: rare tokens (0.06% of positions) never win plain scoring |
| v4 | — → 6751164 | 80 shards × **w10** | 0% prob 0.15 | — | 97–100% ✓ | 0% | ok | w10 warms timing, safe — but not enough |
| v3 | 6746779 → (+6756308 thief) | 80 shards × **w50** | **50% ✓** prob 0.55 | — | 53–98% ❌ | untested | halved | 50× TEACHES timing; the cost is precision (thief below) |
| v5 | 6749313 → 6751180 | re-exposure shards-v5 ×1 — **corrected 2026-09-22: NOT 5× distinct data.** 80 distinct lessons, uneven visits: timed-task 150×, barge 40×, others 30–40× (verified from train.jsonl) | 0% prob 0.000 | — | 97–100% ✓ | PARTIAL (1/6 at 100%, rest 0.03–0.76) | ok | `<yield>` responds to data spread, timing does not |
| v6 | — → 6754413 | shards-v6 × w10 +yield — **corrected 2026-09-22: THREE changes vs v4, not one:** visits barge 150×/timed 100×/others 20–30×, yield weighting ON, and unshuffled sorted order changes the final class seen. Confounded comparison | 0% prob 0.000 (printed; rounded to 3 dp) | — | 97–100% ✓ | weaker than v5 | best (0.10–0.39) | Observed: w10 timing signal (v4: 0.15) gone. Original "DATA×WEIGHT law is DEAD" conclusion is CONFOUNDED — exposure, order and yield-weight changed together; no clean single-variable attribution |
| v7 | 6756309 → 6756310 | 80 shards × **w30** | 0% prob 0.014–0.032 ❌ | **learned**: 1/6 at 100% (prob 0.772), rest 0.047–0.372 | 95.0–99.8% ✓ (mild thief: 4–5 steals/lesson) | 0% | ok | t+5s crosses between w10–w30; t+30s needs ~w50; non-monotonic: t+30s DROPPED vs w10 (frequent t+5s grabs capacity first) |
| v8 | 6756384 → 6756385 | 80 shards × **class-cost 50** | 0% prob 0.000 | 0% prob 0.000 | 99.3–100% but prob **flat 0.250** everywhere | prob 0.250, acc 0 | ~0 | **FAILED — math bug (below). Loss was label-blind; model only learned "never say rare words"** |
| **v9** | 6758553 → 6758554 | 80 shards × **w50 + thief-penalty 10** | 0% prob 0.000 | 0% prob 0.000–0.001 | **99.9–100% ✓ prob 0.958–0.995 (best ever)** | 0% | 0.00–0.13 ❌ | **OVER-SUPPRESSED — "the Monk": perfect silence, everything else dead. The tax fires at every non-rare spot (~1500/lesson) vs the up-push's ~1, so pen 10 swamped w50. Bracket: pen 0 → timing 50% + ~150 steals/lesson; pen 10 → nothing. Next: pen 2** |
| **v10** | 6762715 → 6762716 | 80 shards × **w50 + thief-penalty 2** | 0% prob 0.003–0.013 ❌ (0/6) | alive but weak: prob 0.010–0.352, acc 0/4 | **99.2–100% ✓✓ prob 0.932–0.982** | 0% | 0–19.4%, and `<sil>` steals 21–30 word spots/lesson | **Rare-token thief FIXED** (1–2 steals/lesson, bar was ≲5). But the tax's cliff sits between pen 0 and pen 2: t+30s died anyway. New finding: the dominant error is now `<sil>` eating WORD positions — under-speaking is the live gate's real bottleneck (t1/t2/t4) |
| **v11** | 6765121 → 6765122 (+full exam 6771503, gate 6771504) | w50 + pen 2 + **word-weight 3** | **1/4 ✓ prob 0.426** on 029; 0/4 on 017 (thief 251) | 0/3 firing, prob 0.067–0.317 | 95.0–99.2% ✓ barge; 76.7% ❌ on 017 (thief), 99.8% on 029 | 0% | **0.185–0.387 ✓ (v2-level restored)** | **Speak-up dose WORKED on the exam** — words ~v2-level, partial t+30s recall returned, barge-class silence kept. Cost: the thief re-empowered on the countdown-dense lesson (017: 251 t+30s steals, sil 76.7%). Best exam profile of any run. Gate: PASS 2/5 (below) |

Penalty dial, fully bracketed at w50: **pen 0** → t+30s 50% + ~150 steals; **pen 2** → sil perfect, rare-thief fixed, t+5s weak, t+30s dead; **pen 10** → nothing. The cliff is between 0 and 2; bisecting it buys t+30s recall that the free-run diagnostic already showed needs gold priming to fire live — low value for the gate.

**The bottleneck moved.** Words: v2 (unweighted) 20–48% acc — every weighted run since degraded speaking (v3 halved, v9 zeroed, v10 0–19%). The gate's failing tests (t1 answers late, t2 silent post-barge, t4 early on two-parter) are all SPEAKING-decision failures; the passing ones (t3/t5) are quiet tests. Next lever is word-beat reliability, not more rare-token surgery.

## v11 — the speak-up dose (user-approved direction, 2026-09-21)

**Recipe: w50 + thief-pen 2 + `--word-weight 3` (jobs 6765121 → 6765122).**
The same weighted-CE lever that taught timing, aimed at speaking: word-label
positions (regular-vocab gold, < n_old) get loss weight 3 vs the ~95%
silence background. Disjoint from rare positions by construction. Bracket
plan per cliffs-not-slopes: words don't move on the exam → raise to 10;
sil control breaks → drop to 1.5. Exam is the decision instrument (it
predicts the gate — see live-gate #3); gate rerun only if the exam moves.

## Live-gate result #1: v3 (T1–T5, job 6762718, runs/harness-layera-p1-v3)

**FAIL 1/5** — t1 no answers, t2 never replied after barge, t4 no part2
answer, t5 engaged 22.0 s with background speech. t3 "passes" as a
metric artifact (counts quiet-downs; v3 never goes quiet).

## Live-gate result #2: same v3 ckpt, format-fixed harness (6763486)

Two harness bugs found and fixed along the way (first two rerun
attempts died revealing them):
1. **Cross-test session leak** — the thinker's KV memory, audio buffer
   and beat clock carried from test to test. T2's beats stamped at
   t≈152 s into a 40 s tape (render crash); T3 "babble" (562 word
   beats) was the model continuing T1's stale conversation, not
   responding to backchannels. Fix: `Target.reset()` per test — T3
   babble collapsed to **3 word beats**.
2. **Tape overflow** — window tails put the last beats a step past the
   timeline; render() now pads one tone-length.

**Clean verdict: PASS 2/5** — t3 ✓ (quiet through backchannels), t5 ✓
(false engagement 0.72 s vs 22.0 s pre-fix!). Fails are now MODEL
quality, not plumbing: t1 answered but p50 14.9 s late (sparse word
beats — 9 per test), t2 never spoke after the barge, t4 answered
during part 1 of the two-parter (48 word beats of real text from
chunk 3). Live profile: safe-and-quiet, under-speaks, interrupts
early. Timing over-firing visible but silent (79/9/8 timing beats per
test — no metric cost yet).

The harness now measures the MODEL. Next question: does a better
checkpoint (v10, light thief tax) move t1/t2/t4?

## Live-gate result #3: v10 (6763949) — and the exam→gate mapping validated

**PASS 2/5 (t3, t5 — same as v3) but strictly more muted:** t5 false
engagement 0.0 s (v3: 0.72 s), yet t1 has **no answers at all** (v3:
answered, 14.9 s late), t4 no part-2 answer (v3: spoke during part 1),
t2 unchanged (silent post-barge). Beat census: **0 word beats in ALL
five tests** (v3: ~9/test, 48 real-text beats in t4), 1 timing beat
total, 0 yields.

**The exam predicts the gate.** v10's word-slot sil-steals (21–30/lesson
on the exam) mapped exactly to live muting; v3's sparse words mapped to
late/sparse answers. The 10-minute held-out exam is now a trusted proxy
for the 8-minute gate — future runs need only the exam unless it's
borderline.

Both gates point the same way: the w50 family's remaining defect is
UNDER-SPEAKING, and the unweighted v2 (words 20–48%, sil 99.4%, zero
timing knowledge) has never been live-gated. The T-tests need only
word+silence decisions — v2's strengths. Result pending (6764321).

## Live-gate result #4: v2 baseline (6764321) — the three-way picture

| test | v3 (w50) | v10 (w50+pen2) | v2 (unweighted) | bar |
|------|----------|----------------|-----------------|-----|
| t1 answers | late, p50 14.9 s | **mute (0 beats)** | **mute (0 beats)** | p50 ≤1500 ms |
| t2 reply post-barge | never | never | **replied 1600 ms**; barge→sil 1760 | ≤800 ms |
| t3 backchannels | PASS | PASS | PASS | 0 yields |
| t4 two-parter | spoke during part 1 | mute | 1 beat, no part 2 | part 2 answered |
| t5 background | PASS 0.72 s | PASS 0.0 s | PASS 0.48 s | ≤2.0 s |
| word beats total | ~45 | **0** | 25 (but 10 on t5 — misplacing) | — |

**Verdict: all three pass 2/5. The gate is bottlenecked on speech
PRODUCTION, not style.** v2 is the only checkpoint that ever replied
after a barge (1600 ms — exists, 2× too slow). But v2 also MUTES on t1,
so under-speaking is structural, not a weighting artifact: `<sil>` covers
~95%+ of training ticks, so every model learns a silence attractor it
can't break out of at inference — even the unweighted one. v2 also shows
the opposite failure (10 word beats on t5, engaging background speech).

**Decision point for the next training run:** the lever that fits is
the proven weighted-CE applied to WORD positions (currently weight 1
against a 95% silence background). Doses bracketed per the
cliffs-not-slopes rule. Each dose ≈ 1.1 GPU-h (55 min train + 10 min
exam), and the exam now predicts the gate, so only borderline exams
need the gate rerun.

## Live-gate result #5: v11 (6771504) — exam predicts speaking LEVEL, not PLACEMENT

**PASS 2/5** (t3, t5 @ 0.48 s). Beat census is v2-shaped again: t1 = 0
beats (mute through 10 questions), t2 = 0, t4 = 4 beats with
`premature_answer: true` (spoke during part 1), t5 = 10 beats of
background engagement (0.48 s, under the 2.0 s bar).

**Refined exam→gate rule (two observations each way):** the exam
*catches catastrophic regressions* (v10: sil-dominant exam → fully
mute) and *tracks the amount of speaking* (v11: exam words restored to
v2 level → live speech restored to v2 level: sparse, misplaced) — but
it does **not** predict placement (when to start/stop), which is what
t1/t2/t4 actually fail on. Teacher-forced next-token accuracy cannot
see self-fed timing decisions.

**Caveat on all gate verdicts #1–#5:** an external adversarial review
(CODEX_PLAN.md, 2026-09-21) demonstrated from source that the gate's
scoring marks are offset from the actual stimulus (t1 questions
audible at 17…152 s but scored at 2…137 s), that latency is measured
from question START although questions end ~2.5 s later, that three
placed tones can pass 5/5, and that the t2 800 ms bar is measured from
interruption onset while the training data schedules replies ~2.3–5 s
after onset. Verification confirmed the offsets in source and the
exposure/rounding/freerun claims exactly. **All gate PASS/FAIL counts
are provisional until the marks fix lands; beat censuses remain
informative.**

## Gate instruments REPAIRED (2026-09-22, Codex F1/F2) — rescore pending

The caveat above is now fixed in `evals/harness/` (protocol, metrics,
run, targets) and guarded by `test_gate_regressions.py` (CPU-only,
all cases green):

- **Marks aligned**: t1 scored at the questions' true positions
  (17…152 s); t1 tape lengthened 152→158 s (the last question was
  being truncated at the tape edge). Latency measured from question
  **END** (derived from user_in.wav activity, not a table).
- **Unanswered questions counted**: `n_answered` must be ≥8 of 10 AND
  p50-from-question-end ≤ 1500 ms. A model answering 0 questions now
  FAILS even if one reply was lucky-fast (the old median hid it).
- **t2 reply measured from barge END** (trained schedule is barge end
  +240 ms); stop-fast stays from barge ONSET. The old start anchor
  made every honest model false-FAIL the reply bar.
- **t4 answered_part2 after part-2 END**; talking over part 2 now
  counts as premature (it used to count as answering).
- **Three-tone cheat killed** (regression-test case: tone at each
  question START scores 0 answered, 10 premature → FAIL).
- **Rate bug found while repairing**: user_in.wav (16 kHz) was framed
  at the output rate (24 kHz), stretching derived input events 1.5×;
  input is now framed at its own rate.
- **Stream-contract half-fix in the target**: a single `<sil>` tick no
  longer ends a turn in the live target — ~10 consecutive sil beats
  (~770–800 ms) do, matching training (9,314 of 27,429 `<sil>` ticks
  sit INSIDE assistant speech).
- **Decode knobs live** in `layera/duplex.py`: `--sil-bias` (subtract
  from the `<sil>` logit at decode) and `--max-pieces` word
  continuation ([pad w1 w2] — pieces consecutive after ONE heard
  token, breaker discarded, exactly the training contract; 33.6% of
  training words are multi-piece). `run.py` plumbs both;
  `freerun_diag.py` rewritten free-mode-only with the same knobs (the
  old forced mode fed self word-predictions — retired as flawed;
  probe_p1.py is the true forced reference) and now reports word-onset
  hits (exact / ±1 s), false-speech ticks, continuation pieces.
- `layera/contract_trace.py` added: prints training-format vs live
  token streams for one lesson, position by position, plus the
  52-feats-vs-50-ticks clock table — evidence, not assertions.

**All previously recorded gate verdicts #1–#5 are start-anchored
numbers; a CPU rescore job over the existing recordings
(runs/harness-layera-p1-v2-400step-unweighted/-v3/-v10/-v11) produces
the corrected table. Decode sweep (v11, sil-bias 0/1/2/4/8/16, free
mode, 4 held-out lessons) submitted alongside.**

### Rescored verdict table (job 6777210, repaired instruments)

| test (bar) | v2 unweighted | v3 w50 | v10 w50+tax2 | v11 w50+tax2+ww3 |
|---|---|---|---|---|
| t1 — ≥8/10 answered, p50 ≤1500 ms from question END | **FAIL** 0/10 | **FAIL** 4/10, p50 2200 ms | **FAIL** 0/10 | **FAIL** 0/10 |
| t2 — reply ≤800 ms from barge END | **FAIL** 1520 ms | **FAIL** never replied | **FAIL** never replied | **FAIL** never replied |
| t3 — never yield to backchannels | PASS | PASS | PASS | PASS |
| t4 — answer after part-2 END | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| t5 — ignore background | PASS | PASS | PASS | PASS |

What the honest instruments change vs the old "all three pass 2/5":

- **The pass counts stayed 2/5 but the composition moved** — the old
  t2 PASS was scoring noise; under end-anchoring **nobody but v2
  replies after a barge at all**, and v2's real reply is 1520 ms
  (bar 800; trained schedule is barge end +240 ms). Slow-but-there,
  not passing.
- **v3 is the only checkpoint that answers questions at their ends
  live** (4/10, p50 2.2 s — right structure, too slow and too sparse).
  v10/v11 answer **zero** — v11's exam-level speaking went to wrong
  places (t4 premature beats, t5 background beats), confirming the
  placement problem is not fixed by any dose so far.
- t4 is universally failed: **no checkpoint resumes after the user
  finishes part 2.** Nobody has turn-handback yet.
- Under-speaking remains the top defect (0/10 question ends on three
  of four models) — which is exactly what the sil-bias sweep tests
  without retraining.

## Decode sweep (6777066, v11 ckpt, free-run self-history)

Sil-bias subtracts from the `<sil>` logit at decode — a pure inference
knob, no training. Free-run over held-out lessons with word
continuation ON, 4 lessons, ~900-tick cap (actual ~150–170):
word-onset hits = gold word ticks where the model speaks on the exact
tick; false speech = speaking at non-gold ticks.

| sil-bias | exact word-onset hits (of 83) | false-speech ticks (sum) | `<t+5s>` thief fires |
|---:|---:|---:|---:|
| 0 | 54 | 183 | 0 |
| 1 | 50 | 169 | 5 |
| 2 | 60 | 195 | 3 |
| 4 | 71 | 215 | 3 |
| 8 | 81 | 229 | 5 |
| 16 | ~81 | ~233 | ~2 |

Reading:

- **The knob is a clean monotone threshold**: bias buys exact word
  onsets with chatter (54→81 exact, 183→229 false from 0→8). No
  cliff, no sweet spot on lesson evidence alone.
- **barge_in-017 is the "held-back" profile — and bias unlocks it**:
  0/19 exact word onsets at bias 0 (sil acc 0.677 — mostly silent,
  like the live gate's v11) → 17/19 at bias 8, 19/19 at 16. The other
  three lessons speak early-but-imprecise at bias 0 and sharpen with
  bias. So a model that holds back too much CAN be released by
  decoding alone.
- **Chatter is decode-irreducible**: ~40–58 false-speech ticks per
  ~13 s lesson even at bias 0. On held-out lessons v11 OVER-speaks —
  the opposite of its gate muting. **CORRECTION (same day): this is
  NOT a tones-vs-voice domain gap — the gate prompts were checked and
  are already MiniCPM-TTS voice (t1_q1.wav: envelope CV 0.367;
  rendered by the bake-off's `synth_user_prompts`). The real split is
  SEEN script vs UNSEEN wording: the held-out lessons reuse trained
  utterances (Codex F3), so free-run "over-speaking" there is partly
  script recall, while the gate's questions are unseen phrasing and
  the model falls back to the silence attractor. The clean test for
  that reading is the unseen-wording gate (job after 6777591).**
- Small cost: the `<t+5s>` thief re-appears at bias ≥1 (1–5 fires).
- Sweep gap (recurring LIMIT trap): `--limit 4` with two classes took
  the first 4 matching = all barge_in; timed_task lessons absent.
  LIMIT is total scenarios across classes.

## Live gate under decode bias (6777591, v11 + repaired instruments)

Same checkpoint, same session code, two decode points. bias 0 vs the
original gate #5 isolates the stream-contract change (continuation +
sil-run end-of-turn); bias 4 is the mid-dial point from the sweep.

| test (bar) | v11 bias 0 (contract-fixed) | v11 bias 4 |
|---|---|---|
| t1 — ≥8/10 answered, p50 ≤1500 ms from q-end | **FAIL** 0/10 | **FAIL** 5/10, p50 3120 ms |
| t2 — reply ≤800 ms from barge END | **FAIL** never | **FAIL** 1920 ms |
| t3 — never yield | PASS | **FAIL** (1 yield) |
| t4 — answer after part-2 END | **FAIL** (premature) | **FAIL** (silent at part 2) |
| t5 — ignore background | **FAIL** 5.84 s | **FAIL** 8.96 s |

Findings:

- **The live muting was substantially a DECODE THRESHOLD.** bias 4
  takes question-end answers 0/10 → 5/10 and produces this
  checkpoint's first-ever post-barge reply (1920 ms; bar 800; v2's is
  1520 ms). The model could speak on tones all along — the argmax was
  falling a hair short of `<sil>`→word flips at every beat.
- **The costs arrive exactly as the sweep predicted**: more speaking
  ⇒ background chatter (t5 0.48→8.96 s) and one backchannel yield.
  There is no sweet spot on this dial yet — amount and control move
  together. sil-bias is a REPORTING knob for diagnosis, not a fix.
- **What decoding cannot touch**: reply latency (3.1 s vs 1.5 s bar),
  part-2 turn handback (absent at every bias), and the amount/control
  coupling. Those are training-side (dose/data) — and the next dose
  decision now measures each of them separately with honest
  instruments.
- Contract trace (same job): multi-piece word events present in live
  ([pad w1 w2] shape — `[(4, 8), (9, 8)]` = two 8-piece events);
  952 live features vs 915 training ticks (1.04×, the window
  surplus); beat-vs-tick offset CONSTANT across all three printed
  windows (the per-window step resets — no cumulative drift; the
  printed 2.846 s "delta" is an index artifact: beats include
  continuation pieces, ticks don't). Note: both multi-piece events
  ran to the 8-piece cap — the breaker usually fires but not always;
  consider a soft cap (~4) if runaway pieces show up in M2 voice.

## Unseen-wording gate (6781995) — v11's speaking does NOT generalise

The gate prompts were already TTS voice, so the fair question is seen
script vs unseen wording. Same T-protocol shapes, new questions/bait/
backchannels/parts/background (zero phrase-bank overlap, same user
voice; prompts at runs/bakeoff/prompts-unseen).

| test (bar) | v3 bias 0 — seen → unseen | v11 bias 4 — seen → unseen |
|---|---|---|
| t1 answered (≥8/10) | 4/10 → **2/10**, p50 280 ms | 5/10 → **0/10** |
| t2 reply after barge end | never → 2560 ms | 1920 ms → broke early (720 ms) |
| t3 yields | PASS → PASS | 1 yield → **2 yields** |
| t4 part-2 answer | FAIL → FAIL (premature) | FAIL → FAIL (silent) |
| t5 background | PASS → PASS | 8.96 s → **22.0 s (whole tape)** |

Findings:

- **v11's bias-4 speaking was script recall, not conversation.** On
  unseen wording it answers zero questions and chatters through the
  entire background test. The w50+tax2+ww3 dose taught phrase-level
  replay plus a fragile decode threshold — the exam gains were
  script-level and did not survive new phrasing.
- **v3 keeps the most real behaviour**: halves on unseen questions
  (2/10) but when it speaks it is FAST — p50 280 ms from question end
  (human ~250 ms; bar 1500 ms). The turn-taking TIMING circuit
  exists in v3; coverage (2/10) is what fails. It also stays silent
  through unseen background speech (t5 PASS at bias 0 — the only
  config of six today with t3+t5 both clean AND some t1 speech).
- **Implication for the next dose**: adding more weight to the same
  80 lessons (v12 ww6…) is now the LEAST supported direction — v11
  shows that path buys recall, not generalisation. The evidence now
  favours data-side (more distinct conversations / word-denser
  lessons — Codex F5-3) or capacity (adapter test — F5-2), decided
  with this unseen gate as the selection instrument.

## External reference: "How OpenAI Built GPT-Live" (ByteByteGo, 2026-09-22)

Owner-supplied teardown (interviews with Zahan Malkani + Justin Uberti)
of OpenAI's full-duplex voice system. Integrates as VALIDATION + a
serving-phase map; contains nothing on training technique (our
exposure-bias thread is ours to solve).

Validates our design bets by name:

- **80 ms full-duplex clock, silence-as-token** — Moshi cited as the
  reference ("roughly one frame every 80 milliseconds; silence is
  simply another token"). Exactly our tick loop + `<sil>`.
- **Turn-taking INSIDE the model** — the separate turn-detector of
  generations 1–2 is named as the source of unnaturalness; moving
  endpointing + barge-in into model behaviour is the fix. That is our
  `<sil>`/`<yield>` design, and the article's eval framing ("each
  decision scored like a prediction: endpointing, barge-in") is
  exactly our repaired gate's t1/t2.
- **Delegation = talking/thinking split** — small fast voice model
  `<delegate>`s to a frontier model and keeps chatting. Our
  `<delegate>` token exists but is unimplemented (M2+); the article is
  the design source for that phase: live path audio-only, async path
  for delegation/tools, frontier-model prefill warming, one-RTT
  session setup (WARP), session affinity + managed handoff, p999 tail
  engineering (a p95 event fires several times/minute on a continuous
  stream), capacity in concurrent sessions.

Also useful framing for gate design: "users hear the worst frame" —
per-decision misses (a premature interruption) are the unit of
quality, which is how our per-question answered/premature counting
already works. No changes to Phase-1 plans from this source.

## Capacity test (Codex F5-2, owner-approved) — jobs 6792730, 6814111, 6814321

Question before any next dose: can the 10-row approach (v1 recipe,
~41k trainable params) learn FOUR short conversations at all, vs the
same budget plus a small trainable audio adapter (~1M params,
2048→256→2048 MLP on ear features — the spec's AuT-adapter idea)?
Deliberate OVERFIT probe — capacity, not generalisation. Both arms:
400 steps, unweighted CE, lr 1e-3, same 4 lessons (timed_task-001/004,
barge_in-000/004). Selection instrument = free-run replay
(self-history) on the trained 4, bar ≥90% exact word-onset hits with
chatter <10% of silence ticks; held-out 4 (008/013/012/016)
informational.

**Arm A (rows-only) — job 6792730, checkpoint runs/cap-rows:**

| metric | trained 4 | held-out 4 |
|---|---|---|
| word onsets, exact | **67/68 (98.5%)** | **67/69 (97.1%)** |
| gold `<word>` acc | 57/58–100% per lesson | 89–100% per lesson |
| false speech (share of sil ticks) | 81%, 88%, 44%, 33% | 80%, 69%, 49%, 44% |

Split verdict: the 10 rows learn WHAT to say and EXACTLY WHEN (word
acc ~100%, onset timing near-perfect — and it transfers to unseen
conversations), but they cannot learn to STOP: `<sil>` acc 0.7–6%,
the model babbles lesson words over most of the timeline. The composite
bar (onsets ≥90% AND chatter <10%) FAILS on chatter. Notably worse
silence than even v2 (sil acc 0.944 self-fed): unweighted CE on 4
word-dense short lessons leaves the silence gate unmoved while the
word rows saturate.

**Arm B (rows + adapter) — training completed; free-run re-run
cleanly by repair job 6814321 after a device fix (ckpt
runs/cap-adapter, 1,050,880 adapter params):**

| metric | trained 4 | held-out 4 |
|---|---|---|
| word onsets, exact | 0/14, 0/13, 14/21, 4/20 = **18/68 (26%)** | 13/15, 10/15, 4/21, 0/19 |
| sil acc | **1.000, 1.000**, 0.53, 0.65 | 0.17, 0.34, 0.68, 1.000 |
| false speech (sil ticks) | **0, 0**, 26, 25 | 426, 188, 23, 0 |

**Verdict: NEITHER arm learns the four conversations — composite bar
failed both ways, with complementary failure modes.** Rows-only
places words near-perfectly (98.5% exact onsets, held-out 97.1%) but
cannot learn to stop (chatter 33–88%). The adapter learns to stop —
so well it goes FULLY MUTE on the two timed_task lessons (sil acc
1.000, word acc 0.000, zero false speech) and drags exact onsets to
26%. Small-parameter capacity is NOT the missing ingredient for word
placement; the missing ingredient is free-run speech/silence GATING,
and each lever trades one failure for the other.

Triangulating with the ladder's history:

- v2 (w50, 80 lessons): silence gate works free-run (sil acc 0.944)
  but MUTES — the opposite pole of arm A. So rows CAN learn gating;
  the dose decides which way it falls, and no dose so far has landed
  silence AND speech together in free-run.
- v11 (w50+tax2+ww3): got speech back, but as script recall.
- Both capacity arms trained TEACHER-FORCED and are graded FREE-RUN
  (self-history). Both failing at the same join — the model never
  sees its own chatter during training — points at exposure bias as
  the common cause, not parameter budget and not (only) dose.

This does NOT match either anticipated branch of Codex's completion
criterion ("if the adapter learns and rows cannot, end the rows-only
ladder"): the adapter did not learn either. Capacity, as probed, is
exonerated for word placement (41k params suffice to memorize
placement on 4 conversations) and the adapter is not a free upgrade
(it mutes under this recipe — though unweighted CE on 95%-silence
lessons all but invites the muting solution; a weighted adapter arm
is untested).

Metric bug found reading these logs: freerun_diag's "within ±1s"
counted gold ticks inside their OWN window — vacuously true (a fully
muted run "hit" 14/14). Exact-onset numbers are unaffected; the
event-level near metric is fixed post-hoc (any speech within ±1 s of
the gold tick). Numbers quoted above all use the exact metric.

Gotcha bought twice now: piping job output through `grep` in slurm
scripts silently discards the lines that matter most in a crash (the
exception type/message). Only traceback FRAMES whose path matched
"freerun" survived. No grep on diagnostic pipes from here on.

## Self-history probe (exposure fix) — jobs 6816545 + 6820600 + 6827932, 3 probes COMPLETE

`train_p1 --self-history F` (owner-approved 2026-09-23): each step
free-runs the lesson against itself (no grad, live-exact decode) and —
with probability F — trains on ITS OWN emission tokens as input with
GOLD labels. Audio and targets unchanged; only emission-slot inputs
become on-policy. Own history refreshed every 8 SH-steps per lesson.
Recipe: rows-only, 400 steps, timing-weight 50 (argparse default, as
in every capacity arm), F=0.5, the same 4 conversations and pass bar.

**Probe 1 (cap-sh, unweighted) — FAILS the bar on the mute pole, and
the silence gate finally MOVES:**

| metric | arm A (rows, TF only) | cap-sh (rows + SH 0.5) |
|---|---|---|
| sil acc, trained 4 | 0.007–0.061 | **0.583–0.965** |
| sil acc, held-out 4 | 0.008–0.059 | **0.579–0.926** |
| exact word onsets | 67/68 (98.5%) | **1/68 (1.5%)** |
| chatter (sil ticks w/ speech) | 33–88% | **0–7%** |

Reading: exposure to its own output is the first lever that ever
taught free-run silence at this strength (v2's 0.944 was the prior
best) — the exposure-bias hypothesis is CONFIRMED as the silence
mechanism. But unweighted, recovery-from-chatter degenerates into
"always shut up": on trained lessons the model now mutes where it
should speak (word acc 0–7%). The ladder's pole problem in a new
guise: this lever lands on the OPPOSITE pole from arm A.
NEW wrinkle: a `<t+2s>` thief (up to 144 steals/lesson, held-out
included) — plausibly self-reinforcing, since SH feeds the model's
own wrong picks back as inputs.

Probe 1 mechanics: training ran clean (400/400 steps); both evals
then died at the WORD ONSETS print — the vacuous-metric refactor had
left a NameError (a `py_compile`-invisible runtime break; fixed +
guarded by `evals/harness/test_freerun_metrics.py`, and the repair
job runs it before GPU time). tt-001's per-kind block had already
printed pre-crash, which is where the pole-swap was first seen.

**Probe 2 (cap-sh2, job 6820759) — one-lever delta: + `--word-weight
3`. Speech partially returns; the countdown-token thief returns with
it. Bar still FAILED (25/68 = 37% exact vs 90%):**

| lesson set | cap-sh (probe 1) | cap-sh2 (probe 2) |
|---|---|---|
| barge lessons: exact onsets | 0/21, 0/20 | **13/21, 10/20** |
| barge lessons: sil acc | 0.53–0.65 | 0.57 (both) |
| timed_task: exact onsets | 1/14, 0/13 | 1/14, 1/13 |
| timed_task: sil acc | 0.91–0.97 | **0.34** (`<t+30s>` ×505, `<t+10s>` ×318 steals) |
| chatter | ~0 everywhere | ~0 trained; one held-out at 140 ticks |

First config where speaking and stopping COEXIST (barge lessons: words
at 50–62% accuracy with silence holding at 57%). But on the word-dense
timed_task lessons the countdown tokens — which carry timing-weight 50
in every one of these runs — eat most positions once speech is
re-enabled. This is the SAME thief that broke the ladder's best
speaker (v11: 251 steals on the countdown-dense lesson), now reproduced
in miniature on the 4-lesson rig, and plausibly self-reinforcing under
self-history (the model's own wrong picks are fed back as inputs).

**Probe 3 (cap-sh3, job 6827932, 3h10m) — one-lever delta: +
`--thief-penalty 2`** (the ladder's v9 counter, part of v11's best
recipe). Question: does the thief penalty protect the timed_task
lessons while self-history + word-weight hold the poles?
**ANSWER: NO — the tax re-muted the model. FAILED (4/68 = 5.9%
exact vs 90%), WORSE than probe 2's 37%:**

| metric | cap-sh2 (probe 2) | cap-sh3 (probe 3) |
|---|---|---|
| exact onsets, trained 4 | 25/68 (37%) | **4/68 (5.9%)** — t-001 0/14, t-004 0/13, b-000 4/21, b-004 0/20 |
| exact onsets, held-out 4 | (18/21 best) | 5/70 — only b-016 alive (5/19 exact, 14/19 near, 26 chatter) |
| full mutes (trained) | 0 | **2/4** (t-004, b-004: sil acc 1.000, zero speech) |
| t+30s thief at sil positions | ×505 (t lessons) | **still ×42 on t-001** — the tax did NOT remove it |

The dose-response cliff replicated in miniature: exactly as the
ladder's thief-penalty 2 (v10) killed v3's speaking and timing, pen 2
on the 4-lesson rig suppressed the words the word-weight lever had
just revived — while the thief itself survived free-run. The tax buys
quiet, not precision. One encouraging crumb: b-000 fired its gold
`<t+10s>` correctly (acc 1.000, 1/1) — the first correct timing-token
fire anywhere in the free-run series.

**Probe series verdict (3 probes): the exposure lever genuinely moves
free-run silence (probe 1), and word-weight 3 genuinely revives speech
on top of it (probe 2, the only both-poles coexistence) — but the
config sits on a knife edge: +penalty 2 (probe 3) or −word-weight
(probe 1) collapses to a pole. Dose-response, not a stable fix. The
next move is a decision point (owner): word-weight dose scan (1.5 / 6)
under SH, cut SH's feedback share (F 0.5→0.25 — the mute may be
self-reinforcing: the model's own silence gets fed back as input and
95% of gold labels agree with it), or the data-side fix (Codex F5-3,
word-denser lessons) which attacks the 95%-silence root directly.**

Audition renders (owner request, job 6827931 → FAILED 18:47, refixed):
all three checkpoints free-run the 4 trained conversations with
`--save-picks` (picks survived; renders died), and
`evals/render_model_audio.py` speaks each model's words (factory TTS
voice) at the exact ticks chosen, over the in-slot audio —
`runs/model-audio-cap-rows` (chatterer), `-cap-sh` (muted),
`-cap-sh2` (first coexistence), `-cap-sh3` (re-muted) for the
localhost gallery. 6827931's failure: MiniCPM-o TTS crashed INSIDE its
own model code mid-word (`llm_embeds + hidden_embeds` 2048 vs 2047 —
its 2048-code cap), leaving no wav; `sf.read` then died on the missing
file and `set -e` killed the job. Fix: per-word try/except + one
retry in the renderer — an audition tool must skip one bad word, not
lose the batch.

## Word-dense lessons (data-side fix) — owner-approved 2026-09-24, batch synth-dense1

After the probe series mapped the knife edge (exposure + word-weight
swing between poles, penalty re-mutes), the owner chose the data-side
lever: **the root cause is that ~95% of every training tick is
`<sil>`** — no training dial reliably beats those odds. New lesson
types give the word and silence poles comparable rates:

- **`dense_dialog`** (`gen_dense_dialog`): 7 rapid-fire Q/A exchanges
  from a new hand-written 20-pair bank (`DENSE_QA`, ADR-0002 — no
  closed-API content), 200–400 ms gaps. Measured on the scheduled
  timelines: **71–72% assistant speech**, ~110 s span.
- **`dense_barge`** (`gen_dense_barge`): three 300–600 ms-gap chunks
  of an EXPLAIN text, barge mid-chunk 2/3 from `DENSE_BARGES`,
  clean-yield reply; nothing scheduled past the barge (the renderer
  truncates the chunk in progress). **93–105% assistant speech**.
- Both feed the existing render→filter→gallery pipeline unchanged
  (`synth-render.slurm` gained a `--classes` passthrough, `CLASSES=`
  env; batch `runs/synth-dense1`, per-class 8, seed 907, chatterbox).
- **REALIZED density (grid audit, `evals/grid_density_audit.py`,
  2026-09-24): 12% word / 88% sil — NOT the 40%+ the 72%-speech
  design suggested.** The emission contract is one word label per
  spoken word (~1 tick / 400 ms of speech), so natural speaking rate
  caps word ticks at ~20–24% even at 100% speech density. Honest
  before/after: old batch total 7% word / 92% sil, with timed_task
  deserts at 1–2% and barge_in already 11–15%; dense batch uniform
  11–14%. Net: deserts gone, ~2× overall word rate, 88% sil is
  structural to the labeling contract. CONSEQUENCE: the data lever
  alone probably cannot equalize the poles — the loss side
  (word-weight) multiplies whatever density exists (12%×3 = 36%
  effective in probe 5; ww 6 → 72%). Re-labeling speech-active ticks
  would be a SPEC change — bigger decision, not taken now.
- Timing-token design note: the grid timestamps an assistant onset by
  ITS OWN silence gap, so dense answers carry small-bucket `<t+5s>`/
  `<t+10s>` at EVERY answer onset — dense, correct-position fires
  instead of rare, over-weighted ones. The thief feeds on rarity.
- Filter: 12/16 pass (dense_barge-007 cut_misplaced — barge estimate
  drifted past the realized chunk end; dense_dialog-003 tail_babble
  "sure"; 2 more above). Normal attrition, over-render covers.
- **Probe 5 (dense-sh1, job 6843717, chained after sharding 6843694):
  the probe-2 recipe verbatim (SH 0.5, refresh 8, ww 3, NO penalty)
  on 4 dense lessons — ONE changed variable: the data.** Same pass
  bar; held-out 4 informational.

## Free-run diagnostic (6762966 crashed on a type slip → 6762979, v3 ckpt)

Replays held-out factory lessons in the EXACT training format, twice:
forced (gold history — should match the probe) and free (self-history).
Ear-rate alignment: 52 tokens/4 s window vs 50 ticks (ratio 1.04 — the
sampler's one-feature-per-tick assumption holds, small edge effect).

| lesson | mode | sil acc | t+30s | words | recoveries to sil |
|--------|------|---------|-------|-------|-------------------|
| timed_task-017 | forced | 0.383 (t+30s x675 steals) | 4/4 | 0/17 | — |
| timed_task-017 | **free** | **0.944** | 0/4 | 0/17 | 81 |
| timed_task-029-ish | forced | 0.398 (t+30s x666, t+60s x99) | 2/2 | 0/13 | — |
| timed_task-029-ish | **free** | **0.922** | 0/2 | 0/13 | 110 |

**Verdict — corrected 2026-09-22 (external review caught an overstated
claim; raw log re-checked):**
1. Free-running in the training format is CLEANER than forced (~93%
   silence, zero chatter, 81–110 self-recoveries), but the earlier claim
   "free-run never fires t+30s" was **wrong as phrased**: at the gold
   t+30s positions it scores 0/6, yet it fires t+30s **159× at wrong
   `<sil>` positions** in free mode (61 on 017, 98 on 029 — verified in
   freerun-6762979.out). The "gold primer" mechanism story is NOT
   supported. What survives: free-run theft is ~10× lower than forced
   (159 vs 675+666 per lesson) — self-history reduces but does not
   eliminate the thief.
2. Forced mode here was NOT a true teacher-forced replay for words: the
   diagnostic fed the model's own word predictions back even in forced
   mode (gold word → `gold_id=None` → self-feed, freerun_diag.py:141/:160).
   Forced reference numbers should come from probe_p1 (true
   teacher-forcing) instead; the flawed forced mode is retired.
3. The live loop fed a DIFFERENT format: chat template + "(listening)"
   envelope + processor `<|audio|>` per-tick path. The checkpoint
   learned the bare stream, so in chat-wrap it did the chat-brain
   thing: tried to answer, forever. (This finding stands — see also
   the stream-contract section below: the match was still imperfect for
   multi-piece words.)

**Fix (duplex.py v2 + LayerATarget v2): live now mirrors
prepare.encode_audio exactly — no envelope, 4 s-window ear features
scattered into `<|audio_pad|>` slots, one beat per feature, beats
timestamped so output lands at the right wall-clock offset. Live gate
rerun on the SAME v3 checkpoint = the A/B.**

Cheap decode-side note: in Phase-1 the specific old-vocab word emitted
does not matter (voice is M2; tone = decision), so the gate only needs
WHEN, not WHAT.

## The v8 post-mortem (why class-cost failed)

Intended: rare tokens cost extra BOTH when correct (up-push) AND when
wrongly predicted (down-push). Actual math:
`L = Σ_j c_j·p_j − c_y` ⇒ `∂L/∂z_k = p_k·(c_k − E[c])` — **no y
anywhere**. The "when correct" half (−c_y) is a constant: zero gradient.
So the only learnable content was "suppress rare tokens everywhere",
which is what the exam showed: all timing 0.000, and sil/yield parked at
a context-free 0.250 in every scenario (a learned constant, not a
decision).

**v9 = the corrected idea, as two working pieces:** the 50× up-push
stays in the label-dependent weighted scoring (v3), and the down-push is
a separate `--thief-penalty` that taxes rare-token probability mass ONLY
at positions whose correct answer is not a rare token — exactly the
thief's turf (584/588 sil spots stolen by t+30s in v3).

## The thief (named 6756308, v3 failure-argmax histogram)

v3's "broken silence" was never broken silence — it was `<t+30s>`
OVER-FIRING at silence positions (scenario 029: 584/588 sil failures
were t+30s steals; 014/017/018 same). Timing recall fine; precision
broken. Per-position weights boost only the up-push; the down-push at
silence stays weight 1. This is what v9's tax fixes.

## Gotchas that cost us (do not re-buy)

- **probe-p1.slurm: always pass SHARDS_DIR** — without it the exam
  silently includes TRAINING rows (header prints "training set was 0
  shards"). First v4/v5 probe attempt (6749312/6749314) was discarded
  for exactly this.
- **Env by inheritance, NOT `--export`** — sbatch `--export` splits
  values on commas, silently truncating CLASSES=timed_task,barge_in
  (probe 6746780).
- **Submit from $DL/repo** — relative log paths follow the submission
  cwd.
- **Checkpoint provenance:** each `p1_checkpoint.pt` stores its `args`
  dict — the definitive record of what any run actually was (v2 was
  found hiding in `runs/p1-v1` this way).
- **This setup has CLIFFS, not slopes** — v6 (shards-v6 ×w10) and v9
  (pen 10) both landed timing at printed prob 0.000 (rounded to 3 dp —
  anything < 0.0005; "exactly zero" was an overclaim, corrected
  2026-09-22), and v9's words also died. Treat dials near the cliff
  with bracketing (bisect between the last working and first dead
  dose), not extrapolation — but note the external review's caveat:
  discrete argmax flips can look like cliffs while underlying scores
  move smoothly.
- **Loss normalization caveat (external review, verified):** the
  weighted-CE term divides by the weight sum over ALL positions
  (including audio/ignored positions), while the thief penalty divides
  by graded positions only (`train_p1.py`). Internally consistent
  across runs at fixed recipe, but the cross-term scale between CE and
  penalty is arbitrary — do not interpret `thief_penalty` doses as
  comparable across recipes.
- **"5× data" never meant 5× distinct data** — v5/v6 re-visited the
  same 80 lessons with uneven per-class visit counts (v5: timed 150×,
  barge 40×, others 30–40×; v6: barge 150×, timed 100×, others 20–30×;
  verified from train.jsonl). Any future data-side claims must report
  actual per-class visit counts, not shard-file multipliers.

Checkpoints preserved: `runs/p1-v1-32step`, `runs/p1-v2-400step-unweighted`
(renamed from p1-v1), `runs/p1-v3` … `runs/p1-v8`.

**Cluster outage note (2026-09-25):** all Isambard services offline for
planned essential maintenance (status page: updates after 16:00 BST).
Window overlaps probe 5's retry (6849442, started 15:27 on 09-24,
8 h budget, needed until ~22:30): if maintenance drained the system
before it finished, the job died mid-run again and needs resubmit #3;
if it started after ~22:30, results are on disk. Resolved 2026-09-25:
the job FINISHED at 05:02:49, ~2.5 h before the outage cut access —
no rerun was needed. Verdict below.

## Probe 5 verdict (dense-sh1, job 6849442, 2026-09-25)

Training finished in 05:02:49 under the 8-hour wall (attempt 1's 4-hour
timeout was the only real problem). Pass bar (trained 4): ≥90% exact
word-onset hits AND chatter <10% of silence ticks.

| lesson             | exact onsets    | within ±1 s | word-on-silence chatter |
|--------------------|-----------------|-------------|-------------------------|
| dense_dialog-001   | 5/99  = 5.1%    | 33/99       | 48/798  = 6.0%          |
| dense_dialog-004   | 7/103 = 6.8%    | 74/103      | 69/793  = 8.7%          |
| dense_barge-001    | 3/63  = 4.8%    | 36/63       | 25/384  = 6.5%          |
| dense_barge-004    | 2/39  = 5.1%    | 16/39       | 26/263  = 9.9%          |
| **trained total**  | **17/304 = 5.6%** | 52%       | 168/2238 = 7.5%         |
| held-out total     | 14/291 = 4.8%   | 34%         | 185/2180 = 8.5%         |

**VERDICT: FAIL the onsets bar; PASS the chatter bar.** But the
failure signature changed qualitatively vs probe 3:

1. **Not muted anymore.** Every lesson speaks; chatter sits just under
   the 10% bar on all four trained lessons. Probe 3 had two fully
   silent lessons at 5.9% exact.
2. **Placement, not timing smear, is the failure.** The "within ±1 s"
   numbers look encouraging but are at or BELOW chance: with ~7.5%
   chatter spread uniformly, a 15-tick ±1 s window would catch a stray
   word ~70% of the time (1−0.925¹⁵) — dialog-004's 72% is chance,
   the rest are under. Direct check: on dialog-001 only 15 of the 48
   silence-position words were within ±1 s of any gold onset, so ~25%
   of the model's off-gold speech sits near gold — about what random
   placement against 12% gold density predicts. The model speaks
   roughly half the right TOTAL amount (mid-run self-play: 5.4% word
   ticks vs 12% gold) but its word starts are uncorrelated with gold
   onsets.
3. Yields 0/3 (each dense barge lesson has one); the small-bucket
   timing tokens dense data feeds (t+5s/t+10s) mostly missed at tiny n;
   the t+thief still nips (19 t+10s steals on dialog-001's silence).

Dose-response arc across the probe series: **mute (probe 3) →
under-speaks: right amount, wrong places (probe 5).** Dense data +
word-weight 3 coaxed speech out but taught no onset placement.

Decision tree (pre-registered above) lands between "partial" and "no
change". The pre-registered next dose is word-weight 6 on the dense
rig (~72% effective word share; ~5 h GPU). Caveat for the owner:
placement being uncorrelated means ww6's realistic best case is "right
amount, still wrong places" — if a dose does not sharpen placement,
the remaining suspect is the labeling contract itself (one word tick
per spoken word ≈ 1 tick per 400 ms of speech, structurally capping
word ticks at ~20%), which is a SPEC change, not another dose.

## Probe 6 verdict (dense-sh2 = word-weight 6, job 6865063, 2026-09-26)

Owner-approved next dose, one changed variable vs probe 5: ww 3 → 6.
Completed 06:28:08. Same pass bar.

| lesson             | exact onsets   | within ±1 s | silence acc | word-on-sil chatter |
|--------------------|----------------|-------------|-------------|---------------------|
| dense_dialog-001   | 10/99 = 10.1%  | 76%         | 0.328       | 125/798 = 15.7%     |
| dense_dialog-004   | 2/103 = 1.9%   | 34%         | 0.223       | 66/793 = 8.3%       |
| dense_barge-001    | 0/63 = 0%      | 10%         | 0.984       | 6/384 = 1.6%        |
| dense_barge-004    | 2/39 = 5.1%    | 21%         | 0.437       | 40/263 = 15.2%      |
| **trained total**  | **14/304 = 4.6%** | —        | —           | **237/2238 = 10.6%**|
| held-out total     | 16/291 = 5.5%  | —           | —           | 256/2180 = 11.7%    |

**VERDICT: FAIL both bars — the dose went the WRONG way.**

1. Placement did not sharpen: exact onsets 4.6% (vs 5.6% at ww3, bar
   90%). The dialog-001 ±1s jump to 76% is window saturation, not
   skill — with 15.7% word chatter PLUS a <t+5s> flood, a 15-tick
   ±1 s window catches something almost always.
2. The <t+5s> THIEF exploded: 375+512+0+106 = 993 timing-token steals
   on trained silence ticks (probe 5: 19 on one lesson). Dense data
   made t+5s plentiful and ww6's word pressure pushed mass into its
   neighbor. Silence accuracy collapsed 0.91 → 0.22–0.44.
3. Bistable per lesson: dialog-004/barge-004 FLOOD (t+5s everywhere),
   barge-001 SHUT UP (0.984 sil but 0/63 words). The dose response
   isn't even a slope — lessons fall off the same cliff in opposite
   directions.

Dose ladder complete: ww3 = right amount, wrong places; ww6 = more
wrong words AND a broken quiet pole. **More loss-side dosing is
refuted.** Per the pre-registered tree, the labeling contract is now
the prime suspect: one word label per spoken word (~1 tick per 400 ms)
structurally caps word ticks at ~12–20%, so the grid the model learns
on is ~85% silence with sparse, tick-exact onsets — and cranking word
weight trades placement for noise instead of teaching placement.
Next decision belongs to the owner: re-label per-tick speech (SPEC
change) vs move up the stack and accept Layer-A timing as-is.

## v2 label contract: per-tick speech (ADR-0004, owner-approved 2026-09-26)

**Decision:** re-label the emission contract. Every tick inside an
assistant utterance now carries `WORD` (a new word starts) or the new
`<speak>` (continue current voicing), instead of v1's one word tick per
word with `<sil>` on every continuation tick. Between-utterance gaps
keep true `<sil>` (that is the barge-in surface); timing tokens and
yields keep their slots (only SIL upgrades to SPEAK, so nothing
existing changes meaning).

**Why:** probes 5-6 refuted loss-side dosing on a grid that is ~88%
silence by construction (one word label per ~400 ms of speech). The
model reasonably learned the silence; word weight only traded placement
for noise. Per-tick labels give gradient signal on every spoken tick
and make "when to start" the natural boundary.

**Realized density (grid audit re-run, 12-lesson dense batch):**
word 12.0% + speak 47.7% = **speech labels 59.7% vs sil 39.9%**
(v1: 12% / 88%). Poles near-balanced with zero deserts; timing tokens
unchanged at 0.35%.

**Recipe consequence:** word-weight returns to 1.0 — its only purpose
was compensating v1 label sparsity, which the contract change removes
keeping ww3 would triple an already-balanced class (probe 6 showed
where excess word pressure goes). SH 0.5 + refresh 8 and timing-weight
50 stay. ONE conceptual variable: the labels.

**Metrics (freerun_diag v2):** onsets are now gold SPEECH-RUN starts
(first tick after quiet) — `speech_run_onsets`, unit-tested; new
`SPEECH: P/R/F1` line scores per-tick speech activity; chatter counts
model speech on genuinely-quiet gold ticks. **Pass bar (probe 7):
≥90% exact run-onset hits AND chatter <10% of quiet ticks** on the
trained 4 — same numbers as before, attached to the v2 definitions.

**Probe 7 (dense-pt1)** = probe-2 recipe base (SH 0.5, refresh 8,
timing-weight 50, ww 1.0) on the SAME trained/held dense 4, shards
rebuilt under v2 (`shards-dense2`). Comparable only to future v2 runs;
v1 probe numbers stay the historical record.

## Probe 7 verdict (dense-pt1 = v2 labels + ww 1.0, job 6885910, 2026-09-26)

Completed 03:58:26 (v2 shard build: 12 shards, 1:07). Pass bar (v2):
≥90% exact run-onsets AND false speech <10% of quiet ticks.

| trained lesson   | speak acc | word acc | speech F1 | behaviour |
|------------------|-----------|----------|-----------|-----------|
| dense_dialog-001 | 0.000     | 0.000    | 0.000     | fully muted (355 speak ticks → sil) |
| dense_dialog-004 | 0.000     | 0.000    | 0.000     | fully muted (461 → sil) |
| dense_barge-001  | 0.000     | 0.016    | 0.029     | muted |
| dense_barge-004  | 0.849     | 0.026    | 0.263     | babbles <speak> (166 speech ticks on quiet; its 3/3 exact onsets are saturation) |

Held-out: two muted (F1 0.00/0.00), two partial (0.09/0.16).

**VERDICT: FAIL — re-muted on balanced labels.** The v2 contract
removed the label-imbalance excuse and the model STILL collapsed to
silence at ww 1.0. The dose-response law now reads clean across BOTH
contracts: ww 1 mutes (probes 1, 7), ww 3 speaks at half rate (probes
2, 5), ww 6 babbles + thief (probe 6). The attractor is not label
balance — it is the loss SHAPE: <sil> and <speak> are single tokens
(cheap to master), while the word pole is the whole BPE vocab
(individually rare); the fastest loss reduction is to perfect the
cheap poles and abandon the diverse one.

**Deeper finding (input-evidence):** the model never hears itself —
the in-slot audio carries only user + background. Its own history is
the ONLY self-speech evidence, so under self-history training a
collapsed model keeps feeding itself silence. Step-8 recollect shows
the untrained model says words everywhere (450/450 picks are word
tokens); by end of training it has learned sil mastery and let go of
speech. Self-history amplifies whatever the model currently does.

**Probe 8 candidates (owner decision):** (a) v2 + ww 3 — the only
recipe that ever produced a speaking model, now on labels with 5x the
speech signal; single variable vs probe 7. (b) SH 0 (pure teacher
forcing) — the model always sees GOLD history, so self-speech evidence
is complete while learning; exposure (SH) becomes a later curriculum
step. Not mutually exclusive; (a) has the track record.

## M2 milestone 1: real-time tick parity (m2/parity.py, job 6901381, 2026-09-27)

The docs/04 shadow test, milestone 1: run the exact offline free-run
decode twice per lesson — DRY (as fast as GPU allows) and PACE (tick i
due at t0+(i+1)*80 ms; slips cascade as in a real session) — on the 4
trained dense lessons, dense-pt1 checkpoint.

| lesson          | ticks | pieces | adherence | compute p50/p999 (ms) | parity |
|-----------------|-------|--------|-----------|-----------------------|--------|
| dense_dialog-001| 1398  | 0      | 99.93%    | 55.4 / 65.3           | 0 diff |
| dense_dialog-004| 1453  | 0      | 99.38%    | 55.9 / 110.3          | 0 diff |
| dense_barge-001 |  468  | 12     | 89.96%    | 55.9 / 390.5          | 0 diff |
| dense_barge-004 |  316  | 10     | 88.61%    | 55.8 / 444.0          | 0 diff |

**PARITY: PASS everywhere — the clock does not change what the model
says.** The incremental KV-cached loop paced in real time is
bit-identical to the offline decode every probe was scored with. The
serving skeleton is sound; a human can be wired to it without changing
the model's behaviour.

**LATENCY: the word-continuation loop is the schedule breaker.** One
tick's forward is ~56 ms (24 ms headroom under 80). But a word event
runs its piece loop SYNCHRONOUSLY inside the tick — up to 8 forwards,
p999 390–444 ms — and each overrun cascades ~4 late ticks (88–90%
adherence on the barge lessons). The muted probe-7 model barely emits
words; a SPEAKING model (if probe 8 delivers one) would hit this
constantly, so piece latency is first-class for the Oct 14 goal.
Levers, cheapest first: torch.compile the tick forward (also damps the
~110 ms jitter spikes on dialog-004), CUDA graphs (needs static
shapes/KV padding — real engineering), or streaming pieces across
ticks (design change in the decode contract). Timing the next parity
rerun against probe 8: measure on a checkpoint that actually speaks.

## M2 milestone 1.5: kernel/compile bench (job 6904171, 2026-09-27)

dense_barge-004 (316 ticks, the one trained lesson emitting word
pieces), dense-pt1 ckpt. Parity (dry-vs-paced, same config) = 0
mismatches in every variant.

| variant | p50 / p999 (ms) | vs ref: argmax flips | verdict |
|---|---|---|---|
| batched_mm (ref) | 57.9 / 460 | — | stays the serving kernel |
| eager experts | 169.0 / 1348 | 185/316 (58.5%) | REJECTED: 3x slower AND computationally different |
| eager + compile | 218.1 / 1742 | 155/316 (49.1%) | REJECTED: worse still |

The docs/04 gather+GEMV advice does NOT transfer to this
transformers-on-GH200 stack: the eager expert path is 3x slower at
batch-1 and — the bigger finding — flips over half the argmax picks,
so it is not even the same function (routing/tie-break numerics
differ; whatever the cause, it is disqualified twice over).
torch.compile(dynamic) added overhead on top.

**The budget arithmetic this forces:** single-token ticks fit (57.9
of 80 ms). But a k-piece word costs k+1 extra forwards inside one
tick (pieces sequential; the next ear feature depends on them — no
reordering possible). At speaking rates that is fatal: a model
picking words on 30% of ticks at ~2 pieces/tick averages ~91 ms per
80 ms tick — the loop falls behind permanently. If probe 8 produces a
speaking model, decode speed becomes THE M2 problem. Lever hierarchy
then: (a) measure real piece stats on the speaking model (may be ≪
the fear), (b) live piece cap (behaviour change, measurable), (c)
decode speedup — CUDA graphs (~1.3-1.5x, likely not enough alone),
fp8/KV quantization, (d) accept bounded lag during self-speech with
the catch-up arithmetic (57 ms decode vs 80 ms arrival recovers 23
ms/tick) — audible as ~0.4 s reaction lag after long words, which
lands exactly on barge-ins. Next measurement therefore rides on probe
8's checkpoint.

## Probe 8 verdict (dense-pt2 = v2 + ww 3.0, job 6901314, 2026-09-27)

Same rig as probe 7; ONE variable: word-weight 1.0 -> 3.0 (the muting
endpoint probe, owner strategy 2026-09-27). 400 steps, final loss
0.9060 (probe 7: 1.2756 — ww3 pushed further down, mostly by feeding
the speak/word poles).

| trained lesson   | speak acc | word acc | speech F1 | chatter |
|------------------|-----------|----------|-----------|---------|
| dense_dialog-001 | 0.304     | 0.051    | 0.177     | 221 ticks |
| dense_dialog-004 | 0.104     | 0.029    | 0.161     | 125     |
| dense_barge-001  | 0.062     | 0.016    | 0.062     | 30      |
| dense_barge-004  | 0.151     | 0.051    | 0.157     | 55      |

Held-out F1: 0.046 / 0.000 (dialog-005 fully re-muted) / 0.196 / 0.070.

**VERDICT: FAIL the pass bar — and the muting question is ANSWERED.**
ww3 revives speech under v2 (speak acc 0.000 -> 0.30 on dialog-001);
the model now CAN speak. What it cannot do is PLACE speech: word acc
0.02-0.05 everywhere (still ~random), exact run-onsets 3/5, 3/5, 1/4,
1/3 (bar: >=90%), chatter 9-13%+ of ticks on the revived lessons (bar:
<10%). The failure mode moved from "won't speak" to "speaks at the
wrong times" — per the owner's endpoint rule, the ladder SHIFTS TO
PLACEMENT.

Dose law, final form (v2 labels): ww1 -> mute (probe 7); ww3 ->
partial revival, misplaced (probe 8); (v1: ww3 spoke at the right
rate with random placement — probe 5; ww6 babbled — probe 6).
Mid-training arc (SH recollections): all-word babble at step 8 ->
dithering mid-training -> late collapse on barge-004 (53/304) — the
same bistable arc, softened.

**Probe 9 (dense-pt3) = v2 + ww3 + SH 0** — single variable vs probe
8 (self-history 0.5 -> 0). Rationale: the strongest PLACEMENT
hypothesis is context corruption — the model's own history is its
only self-speech evidence, and under SH 0.5 half the steps train
against its own dithering (the recollect arcs show it). Teacher
forcing gives complete, correct context while the placement mapping
forms; exposure becomes a later curriculum step. Pass bar unchanged.

## Probe 9 verdict (dense-pt3 = v2 + ww3 + SH 0, job 6905339, 2026-09-27)

ONE variable vs probe 8: self-history 0.5 -> 0 (pure teacher forcing —
the placement probe). 400 steps in 1:05 (SH recollection free-runs
were most of the old wall-clock). **The placement battle is DECIDED:
teacher forcing taught placement, and it generalizes.**

| lesson (trained) | word acc | exact run-onsets | sil acc | F1 |
|------------------|----------|------------------|---------|----|
| dense_dialog-001 | 0.697    | **5/5**          | 0.165   | 0.202 |
| dense_dialog-004 | 1.000    | **5/5**          | 0.018   | 0.208 |
| dense_barge-001  | 0.984    | **4/4**          | 0.019   | 0.250 |
| dense_barge-004  | 0.974    | **3/3**          | 0.022   | 0.225 |

Held-out word acc 0.74–0.98 (dialog-002 0.742 with 1/4 exact onsets;
others 5/5, 3/3, 3/3). First probe ever with real, exact, generalized
onset placement — the input-evidence prediction held: give the model
complete, correct context and the placement mapping forms.

**The failure inverted: chatter.** The quiet-discipline collapsed —
gold <sil> acc 0.006–0.357 (was 0.75–0.94), gold <speak> acc ~0.00
(the model never learned to CONTINUE with <speak>; it launches new
words instead), false speech on 261–786 quiet ticks per lesson,
precision ~0.11–0.14. Plain words: it now says the right words at the
right word-ticks, AND babbles words through every gap. Under clean
gold context the ww3 word pole (3x weight, high-entropy, now
placeable) overran the two cheap poles; <sil>/<speak> precision was
the price.

**Reading: the control axes are SEPARATING.** Muting (probe 7),
placement (probe 9 — solved by context quality), and quiet-discipline
(now the sole remaining battle) are distinct mechanisms, each movable
by its own lever. Probe 9's chatter is the expected phase-1 artifact
of the exposure curriculum: the model has never SEEN its own words in
context (SH 0), so it has no self-consistent "after I speak, hush"
policy. The curriculum's step 2 is exactly that experience.

**Probe 10 (dense-pt4) = curriculum step 2: init from dense-pt3,
+SH 0.5, 200 steps.** Needs --init-ckpt resume support in train_p1
(added this session). Hypothesis: exposure AFTER placement forms
preserves the onsets while re-teaching silence from lived context —
SH corrupted probes 7/8 because speech wasn't learned yet; the order
was the problem, not the mechanism. Pass bar unchanged: >=90% exact
onsets AND <10% chatter.
Decode-side companion (same job batch): signed --sil-bias sweep on
dense-pt3 (negative = add to the sil logit = suppress chatter) to map
the placement/chatter tradeoff available at decode time without any
training.

## Decode-side sweep on dense-pt3 (job 6906393, 2026-09-27)

Signed --sil-bias (negative = add to the <sil> logit = suppress) on
the 4 trained lessons: -0.5/-1.0 leave chatter essentially unchanged
(onsets intact); -2 trims ~5-20% and costs the first onset; -4 cuts
chatter 20-40% but recall collapses (0.90 -> 0.46 on barge-001) and
onsets degrade. **The chatter is not a marginal-logit artifact — no
decode-time band-aid reaches the bar.** Confirms probe 10's
curriculum (training-side exposure) as the only real fix.

## Probe 10 verdict — curriculum step 2 failed BOTH ways (job 6906391, 2026-09-27)

dense-pt4 = init from dense-pt3 rows (probe 9's solved placement) +
SH 0.5 for 200 steps: the model's first lived exposure to its own
speech, supposed to graft quiet-discipline onto perfect placement.
Pass bar unchanged: >=90% exact run-onsets AND <10% chatter.

**FAILED, and lost probe 9's win on top:**

| lesson | probe 9 exact onsets | probe 10 exact onsets | chatter (false speech / quiet ticks) |
|---|---|---|---|
| dialog-001 (trained) | 5/5 | **0/5 — fully mute** | 1/443 (mute, not disciplined) |
| dialog-004 (trained) | 5/5 | 3/5 | 179/332 = 54% |
| barge-001 (trained) | 4/4 | 1/4 | 34/160 = 21% |
| barge-004 (trained) | 3/3 | 1/3 | 56/137 = 41% |
| dialog-002 (held) | 1/4 | 2/4 | 216/345 = 63% |
| dialog-005 (held) | 5/5 | **0/5 — fully mute** | 0/398 (mute) |
| barge-002 (held) | 3/3 | 1/3 | 32/98 = 33% |
| barge-005 (held) | 3/3 | 0/3 | 82/160 = 51% |

Speech precision 0.00–0.15 everywhere. Two lessons latched the mute
attractor (gold <speak> acc 0.000, all-<sil>); the rest kept the
chatter attractor. Neither lesson of the curriculum's premise
("exposure after placement forms is gentle") survived.

**Why (training log): the self-exposure source itself was the
poison.** SH recollection re-runs the CURRENT in-training model every
~8 steps to build histories, and its picks flip-flopped between the
two attractors: early recollections ~all-chatter (dialog-001:
1275–1285 of 1344 ticks speech-first), step ~152 fully MUTE (2 of
1344), step ~176 chatter again (390/450), step ~184 mute (2/304).
Training thus saw contradictory self-histories — "my history is
constant chatter, gold says hush" and "my history is silence, gold
says talk" — and the placement mapping eroded under them. Loss looked
healthy (3.56 -> 0.85): the usual loss-shape trap, cheap sil/speak
tokens dominating the average.

**Verdict: the bounded ladder's curriculum step is falsified. Per the
owner endpoint rule (probe 8 bullet): redesign, not more probes of
this shape.** The two decisive facts for the redesign: (1) teacher
forcing gives perfect placement but no self-model — the model has
never seen a correct "after I speak" history (probe 9); (2) raw
self-exposure poisons training because the exposed histories are the
two degenerate attractors, not graded mistakes (this probe). Redesign
target = the EXPOSURE DISTRIBUTION: construct self-history from the
GOLD speech schedule (gold words as word-tokens at gold ticks, gold
silences as targets) instead of recollected free-run — decode-
consistent context ("my last word was X, the user is talking") with a
clean distribution, and no recollection loop (faster runs). Fallbacks
if that half-works: explicit speaking-state input channel; decode-side
min-gap rules as an audition stopgap only (constant sil-bias already
ruled out).

## Probe 11 verdict — corrupt-gold exposure: discipline moved, placement still pays (job 6911469, 2026-09-28)

dense-pt5 = init dense-pt3 rows + SH 0.5 with --sh-mode corrupt (gold
histories with bounded decode-realistic corruption: word-first chatter
bursts in quiet stretches, word starts bleeding early; labels gold).
Ran in 38 min — corruption needs no recollection loop (probe 10 took
3h20). Loss 0.94 -> 0.63, healthy arc.

**FAILED the pass bar, but the axes moved in a lawful way:**

| lesson | probe 9 onsets | probe 11 onsets | probe 9 sil acc | probe 11 sil acc | probe 11 false speech / quiet |
|---|---|---|---|---|---|
| dialog-001 (tr) | 5/5 | 2/5 | 0.006 | **0.510** | 415/443 |
| dialog-004 (tr) | 5/5 | 3/5 | 0.357 | **0.639** | 274/332 |
| barge-001 (tr) | 4/4 | 1/4 | ~0.1 | **0.762** | 104/160 |
| barge-004 (tr) | 3/3 | 1/3 | ~0.2 | **0.504** | 125/137 |
| dialog-005 (held) | 5/5 | 0/5 (near-mute) | — | 0.912 | 42/398 |

Quiet-discipline genuinely improved (sil acc 0.5-0.9 vs probe 9's
~0.0-0.36; dialog-005 held nearly silent with only 42 false-speech
ticks) — the corruption samples DID teach "I just babbled -> hush".
But exact placement decayed again (5/5 5/5 4/4 3/3 -> 2/5 3/5 1/4
1/3), gold word acc fell (0.70-1.00 -> 0.30-0.55), and chatter stays
far over the <10% bar. <speak> acc ~0 everywhere — still never
continues, only launches.

**The three-point law (probes 9/10/11): the placement mapping lives
EXACTLY in the clean-gold history distribution.** Any corruption of
that history during training — recollected attractor garbage (10) or
graded synthetic mistakes (11) — erodes it, and the discipline gain
trades against it roughly 1:1. Training-side exposure has now been
tried in its two pure forms; both pay for quiet-discipline with
placement. The bottleneck looks STRUCTURAL, not distributional: one
softmax head + one embedding-delta carries both skills, and the ww3
word pole keeps the speech-rate prior high no matter what the history
says.

**Stopgap landed in parallel (owner-approved):** dense-pt3 + decode
rules (min-gap 4 ticks + user-energy gate, hold 3 — gold turn gaps
are >=400 ms so no legit onset is blocked) — onsets 5/5 5/5 4/4 3/3
within +-1s (exact ticks blur 1-2), chatter cut 3-10x (786 -> 162
worst lesson), total words 297-887 -> 60-188 per lesson. Audition on
the gallery: runs/model-audio-dense-pt3-gated (rules are a demo patch
— hand-coded turn-taking, not learned). Operational note: TTS
overlay dir is pylibs-minicpm (no "o"); two jobs died on a wrong name
before the render-only fix (6911492 -> 6911511).

## Gate 1 verdict (2026-09-28, job 6912904 dl-gate1) — a real muting dial, but its scores don't rank gold onsets above chatter

Owner decision after probe 11: separate quiet-gate head on the FROZEN
dense-pt3 trunk (placement untouchable; discipline in its own 528,897
params). train_gate.py: binary "should anyone be speaking this tick?"
on the same per-tick hidden state that feeds the word head, plain
balanced CE, half the samples on probe-11 corrupt-gold histories.
Decode (freerun_diag --gate-ckpt): a word-FIRST pick counts only if
the model proposes it AND gate P(speech) >= threshold.

Training is healthy: acc 0.990, loss 0.038 at step 200 (335 s), pred
rate 0.59 vs gold speech 0.59 — no bias. First submission (6912797)
OOMed at 60 s: batched_mm gathers per-token expert weights, a single
42.89 GiB alloc at training seq lengths; eager experts fixed it.

Threshold sweep (trained lessons; chatter = false-speech ticks /
quiet ticks, onsets = gold speech-run starts):

| lesson (quiet ticks)  | 0.35 chatter | 0.5  | 0.65 | exact 0.35/0.5/0.65 | +-1s 0.35 |
|-----------------------|--------------|------|------|---------------------|-----------|
| dense_dialog-001 (801)| 0% (MUTE)    | 0%   | 0%   | 0/5, 0/5, 0/5       | 0/5       |
| dense_dialog-004 (797)| 44%          | 30%  | 21%  | 1/5, 1/5, 1/5       | 4/5       |
| dense_barge-001 (387) | 28%          | 16%  | 8.8% | 2/4, 2/4, 2/4       | 4/4       |
| dense_barge-004 (265) | 33%          | 24%  | 17%  | 0/3, 0/3, 0/3       | 3/3       |

Held-out at 0.5: dialog-002 MUTE (chatter 1%, onsets 0/4, +-1s 1/4);
dialog-005 18% (0/5, 3/5); barge-002 14% (1/3, 3/3); barge-005 MUTE
(0%, 0/3, 0/3). Pass bar (>=90% exact onsets AND <10% chatter):
FAILED everywhere — no threshold passes both halves on any lesson,
though barge-001@0.65 passes the chatter half alone.

Reading:
- The gate IS a learned muting dial: baseline pt3 chatter on
  dialog-001 was 786/801 = 98% of quiet ticks; the gate takes it to 0%,
  and the still-talking lessons drop 2-3x. Cheaper and smoother than
  the hand rules on barge lessons.
- But the dial moves onsets and chatter TOGETHER per lesson — there is
  no operating point that keeps onsets and kills chatter, because the
  scores are wrong, not just the cut: on dialog-001/dialog-002 the
  gate blocks EVERY word-first pick at the loosest threshold (0.35),
  i.e. gold onsets score <0.35 while chatter states on OTHER dialog
  lessons score >0.35. The score does not rank "real turn onset" above
  "chatter" on self-history states.
- Root cause is the same off-distribution trap, one level up: the gate
  only ever saw gold histories and BOUNDED synthetic corruptions.
  Decode visits pt3's own compounded states — e.g. dialog-001's first
  gold onset arrives under an all-<sil> self-history, a state whose
  training labels were always 0 (quiet stretches) but whose true label
  here is 1. The trap follows the training-STATE distribution, not the
  parameter grouping — four-point law now (9/10/11/gate1): whatever
  you train, states off the training distribution break the skill.
- freerun_diag now prints "gate P at gold onsets: tick:P ..." per
  lesson to verify this tick-by-tick.

Pre-agreed branch for half-works: audition render (dl-dgaud) — arm A
gate-only 0.35 (the literal render; dialog-001's mute is audible),
arm B gate 0.35 + approved stopgap rules (min-gap 4, user-gate, hold
3), both full TRAIN4 lessons with picks + TTS into
runs/model-audio-dense-pt3-gate035 / -gate-combo.

Candidate next experiment (owner decision): train the gate on states
pt3 actually VISITS — its own free-run rollouts on TRAIN4 with gold
labels attached (DAgger-style, ~40 min) — so the all-sil-history-at-
onset state is in-distribution with the right label. This tests the
diagnosis directly; if scores separate there, Layer A has its recipe.

## GATE ROUND 2 (DAgger) — verdict (2026-09-28, job 6916784, 33:36)

Recipe: free-run TRAIN4 exactly as decode does with gate1 ACTIVE at
0.35, record the hidden state at every word-first proposal tick with
its GOLD label (buffer: 2400 states, 61.88% speech, pos_weight 0.6),
train the gate on that buffer mixed with gate1's gold/corrupt stream
(acc 0.990, buf acc 0.957). Sweep identical to gate1 for direct
comparison. Chatter numbers below are the LEGACY metric (job ran
pre-v3; v3 reads somewhat lower, direction unchanged).

| threshold | dialog-001 | dialog-004 | barge-001 | barge-004 |
|---|---|---|---|---|
| 0.35 | 5/5 exact, 81% chatter | 5/5, 84% | 2/4, 44% | 3/3, 74% |
| 0.5  | 5/5 exact, 77% chatter | 5/5, 69% | 3/4, 31% | 3/3, 57% |
| 0.65 | 5/5 exact, 78% chatter | 5/5, 65% | 0/4, 14% | 1/3, 26% |

HELD @0.5: dialog-002 1/4 exact (4/4 near; only 1 of 4 onsets even
got a word-first proposal — base-model proposal failure, not gate),
dialog-005 4/5, barge-002 2/3, barge-005 2/3; chatter 52-82%.

- **The diagnosis was right, and the failure moved walls.** Gate P at
  gold onsets: 0.66-1.00 everywhere it was <0.35 in gate1 (dialog-001:
  1.00/0.77/0.99/0.77/1.00 at 0.35 — full mute GONE). Exact onsets at
  0.5 on TRAIN4: 16/17 (94%, bar-passing; gate1's best anywhere: 3/16).
- **But quiet-side discipline regressed to the other wall**: chatter
  31-77% trained, 52-82% held (bar: <10%). Threshold is nearly flat on
  dialog lessons (0.35->0.65: 81->78%) — this is not a tuning problem.
  Round 2 traded the mute wall for the chatter wall; the pass bar is
  still failed on the chatter half.
- Why: on-policy states fix RANKING (the states decode visits now
  carry the right labels), but 2400 buffered states vs 529k gate
  params is memorization territory (buf acc 0.957), and the quiet side
  does not generalize across lessons. Gate2 also reveals the scale of
  the base model's speech-rate prior: it proposes a word on ~91% of
  quiet ticks (dialog-001@0.5: 614 accepted + 114 blocked) and the
  gate passes ~84% of quiet-tick proposals.
- Gate1 audition record (job 6916786, gallery batches
  model-audio-dense-pt3-gate035 / -gate-combo): arm A gate-only 0.35
  = half-mute + residual chatter; **arm B gate1 + stopgap rules
  (min-gap 4, user-gate hold 3) = FULL MUTE, 0/21 exact** — the rules
  compound gate1's over-blocking instead of rescuing it. Stopgap is
  dead as a gate1 patch.
- Composite probe (job 6922786, RUNNING at write time): min-gap
  divides word-first chatter by ~gap (accepted word re-arms
  last_onset; continuation pieces do NOT consume tick slots, so gap
  arithmetic holds). Gold speech-RUN gaps are >= 93 ticks on every
  lesson, so gaps 8/16 cannot clip a real response start; the only
  risk is a chatter accept sitting <gap ticks before a gold onset
  (exact -> near). Arms: TRAIN4 gap 8 (save-picks) / gap 16, HELD4
  gap 8; all thr 0.5 + user-gate hold 3. If chatter lands <10% with
  >=90% exact, gate2+rules is the first full-bar decode config.

## GATE2 COMPOSITE (min-gap + user-gate) — verdict (2026-09-28, job 6922786)

gate2@0.5 + min-gap 8/16 + user-gate hold 3. All thr 0.5.

| arm | dialog-001 | dialog-004 | barge-001 | barge-004 |
|---|---|---|---|---|
| TRAIN gap 8 | 1/5 exact, 13% chatter, R 0.19 | 1/5, 33%, R 0.50 | 0/4, 6.5%, R 0.05 | 0/3, 9.4%, R 0.21 |
| TRAIN gap 16 | 1/5 exact, 6.9%, R 0.11 | 1/5, 25%, R 0.36 | 0/4, 2.3%, R 0.03 | 0/3, 3.8%, R 0.03 |
| HELD gap 8 (all 4 lessons) | 0/15 exact, chatter 5.2-9.7% (all under bar), near 11/15 | | | |

- **The chatter half delivered; the responses died.** Exact onsets
  collapsed to 2/17 trained, 0/15 held, and recall collapsed with them
  (dialog-001 R 0.929 -> 0.192). Mechanism: min-gap counts from EVERY
  accepted word, and gold words inside a response arrive every ~4-5
  ticks — gap 8 blocks the response's own follow-up words, the fed-
  back <sil> diverges the trajectory, and the utterance dies after one
  word. Starts survive (5/5 near on dialog-001); the speaking doesn't.
- Also: gap 8 only halved dialog-004's chatter (553 -> 263), not the
  projected /9 — blocked re-picks come back after the gap because the
  prior re-asserts. On dialog lessons the composite fails BOTH halves
  at once. HELD chatter passes everywhere (5.2-9.7%) — but 0/15 exact.
- **THE DECODE DIAL IS EXHAUSTED.** Four configs now mapped end to
  end: gate1 = mute; gate2 = chatter; gate2+rules = strangled
  responses; rules-alone = audible compromise, never bar-passing. The
  four-point law at decode level: you cannot rule your way out of a
  wrong prior. Remaining levers are training-side (gate round 3:
  gate2-in-loop buffer, quiet-side rebalance) or acceptance (best
  audible config into M2 — gate2@0.5 + user-gate: near-perfect
  onsets, murmur between turns bounded live by the energy gate).

## GATE ROUND 3 (12-shard DAgger) — verdict (2026-09-29, job 6952939)

Buffer: 10,119 states from ALL 12 training lessons (60.6% speech),
rollout with gate2 active @0.5, buffer batch 512; trained to acc
0.985, buffer acc 0.869 (round 2: 0.957 on 2400 states — with 4x the
states the gate is NOT memorizing; it still fails). First submission
timed out at 2h (12-min/lesson rollout; walltime now 4h).

| threshold | dialog-001 | dialog-004 | barge-001 | barge-004 |
|---|---|---|---|---|
| 0.35 | 0/5 exact, 28% chatter | 0/5, 60% | 2/4, 68% | 1/3, 50% |
| 0.5  | 0/5 exact, 18% chatter | 0/5, (692 blk) | — | — |
| 0.65 | 0/5 exact, 8.6% | 0/5, 6.4% | 1/4, 26% | 0/3, 26% |

HELD @0.5: exact 0/4, 2/5, 0/3, 0/3; chatter 21-28%. Gate P at gold
onsets: mostly 0.01-0.35 (recognition GONE — round 2 had 0.66-1.00).

- **Round 3 landed BETWEEN the two walls: partial hush, lost
  placement.** Three rounds = three operating points on ONE frontier:
  gate1 (gold states) = mute pole; gate2 (4-lesson DAgger) = chatter
  pole with perfect placement (16/17); gate3 (12-lesson DAgger, 10k
  states) = middle, good at neither. The recipe family does not
  break the trade-off — it moves the operating point. FIVE-POINT LAW
  (probes 9/10/11, gates 1/2/3): the aux gate reads the SAME trunk
  state the word head misreads; "real turn-start vs chatter" is not
  separable from that state alone, whatever the training recipe.
- v3 metric note: at 0.65 the genuinely-quiet half nearly passes
  (trained 1.9-6.0%, held 3/4 lessons under bar) but R 0.15-0.46 and
  0-2/15 exact — the polite-but-nearly-mute pole. Not usable.
- **Consequence for the plan**: the murmur will not be fixed inside
  this gate family. The law's own remedy is state-coverage = MORE
  LESSONS = the scaling run (layer 3) — which the allocation's
  remaining hours are for anyway. Demo config stays gate2@0.5 +
  user-gate (perfect placement, murmur bounded live by the energy
  rule). One genuinely-new cheap hypothesis remains: gate input =
  trunk state + user-audio energy (the human cue the current gate
  never sees) — round 4, ~1 GPU-hr, optional.

## SCALE1 (the scaling run) — verdict (2026-10-01, job 6971323, 5h55m)

Coverage 33x on the UNCHANGED probe-9 recipe (v2 labels, ww3, SH0):
131 filter-passed training lessons (111 dense_barge + 20 dense_dialog;
240 rendered, filter passed 141) x 2000 steps (~15 passes/lesson),
loss 5.2 -> 0.74. Held-out free-run UNGATED (self-history, the same
instrument as every probe verdict), on the same 4 held lessons the
closing diagnostic picks:

| lesson | exact onsets | within +-1s | false speech (quiet ticks) |
|---|---|---|---|
| dense_dialog-116 | 3/5 | 5/5 | 249/452 = 55% |
| dense_dialog-118 | 2/5 | 5/5 | 330/442 = 75% |
| dense_barge-113 | 0/3 | 1/3 | 86/141 = 61% |
| dense_barge-114 | 0/4 | 0/4 | 0% (FULL MUTE, 0 words, 271 sil-steals) |

**VERDICT: FAIL both bar halves — 5/17 exact (29% vs >=90%), false
speech 41-75% (vs <10%).** 33x more lessons did NOT break the
five-point frontier: the run lands BETWEEN the poles lesson-by-lesson
(3 chatter lessons + 1 mute lesson in the same checkpoint) — the same
one-frontier trade the gate rounds mapped, now at scale. Notable:
+-1s placement is 10/10 on held dialog lessons (the timing circuit
scales; it is the speech/silence discipline that does not).

Run postmortem (chain bugs, all fixed + committed):
- shard stage inherited prepare's default --limit 8 -> first train ran
  on 8 lessons, cancelled, fixed via --per-class 999 (9cc4213).
- filter collapsed dense_dialog: 19/120 round 1 (tail_babble on single
  MiniCPM clips; lesson WER fine 0.04; NO culprit voice — base-rate
  babble x dialog's many assistant clips). Re-roll rounds recover only
  ~+5/roll (round 2: 24, round 3: 29) — re-roll lever EXHAUSTED.
  Queued fix if dialog coverage matters: per-clip ASR check + retry
  inside render.

Next per plan: gate DAgger on the scaled trunk (evals/scale1-gate1.slurm,
UNGATED 8-lesson rollout — can the 33x states split turn-start from
chatter where dense-pt3 states could not), then RL-probe 1 (head-DAgger).

## GATE ROUND 1 ON SCALED TRUNK (scale1-gate1) — verdict (2026-10-01, job 6975066, 1:50)

UNGATED DAgger on runs/scale1-pt1: 8-lesson rollout (4 dialog + 4
barge), 4,374 word-first proposal states (59.8% speech), gate 529k
params, 200 steps, acc 0.988 / buffer acc 0.930 (not memorizing).
Sweep identical to gate rounds 1-3; held = the same 4 lessons as the
scale1 ungated freerun.

- Trained lessons: SAME per-lesson pole-mixing as the base model — one
  lesson full-mute at every threshold (0 speech, 0/5), two near-mute
  (R 0.01-0.06), one chatter pole (R 0.71, onsets 2-3/4 exact, false
  speech massive). Threshold barely moves anything (0.35->0.65: R
  0.709->0.727 on the chatter lesson).
- HELD @0.5: 0/17 exact onsets total; 2 lessons fully mute, 1 chatter
  (222 false-speech ticks), 1 near-mute. VERDICT: FAIL — bar v2 needs
  >=90% exact AND <10% false speech; got 0% / mixed.
- **The five-point law EXTENDS TO THE SCALED STATES: 33x coverage does
  not make trunk-state-only separation possible for this gate family.**
  Gate P at gold onsets is bimodal per lesson (1.00 on lessons where it
  recognizes, 0.02-0.09 where it doesn't) — a lesson-level property,
  not a state-level one.
- Standing conclusions after scale1 + gate1-on-scale1: (a) placement/
  timing scales with coverage; (b) speech/silence discipline does not,
  via recipe-scaling OR gate-DAgger on these states. Remaining levers
  per plan: RL-probe 1 head-DAgger (word-HEAD rows on dagger states —
  different head than the gate), RL-probe 2 segment GRPO, gate input
  += user-audio energy (round-4 hypothesis).

## RL-PROBE 1 (head-DAgger on scaled trunk) — FAIL, with a decisive finding
(2026-10-02, jobs 7001927 -> crash -> 7003278; panel log in repo logs
dl-sc1hd1-7003278.out)

Recipe: 8-lesson raw-decode DAgger rollout on scale1-pt1 (4,374
word-first-proposal states saved to runs/dagger-states/scale1.pt; 556
word-gold = 12.7%, 3,818 control-gold), train ONLY the 11 duplex head
rows, full-vocab CE, 300 steps (~12 s). UNGATED freerun on TRAIN4 +
HELD4. Job 7001927 crashed at train start (fp32 states @ bf16 W —
fixed bdaae6e) AFTER saving the buffer; relaunch resumed via --states.

HELD4 paired vs the scale1 closing freerun (layera/panel.py, FAIL 3
lesson-gates):
  lesson          scale1 (base)     probe-1           gate
  dialog-116      3/5 exact, 5/5±1s 0/5, 2/5          FAIL (all 3)
  dialog-118      2/5 exact, 5/5±1s 0/5, 2/5          FAIL
  barge-113       0/3, 1/3 (chatter)1/3, 2/3          PASS (marginal)
  barge-114       0/4 FULL MUTE     2/4, 4/4±1s       FAIL chatter 54>0
TRAIN4 (no ungated baseline; report-only): dialog-001/010 0/5, 2/5±1s
(<speak>-chatter pole); barge-001 2/4; barge-004 3/4, 4/4±1s, and the
ONLY lesson still emitting words (7 continuation pieces).

**The finding: the <speak> collapse.** On 7 of 8 lessons continuation
pieces went to ZERO and gold <word> accuracy to 0.000 with <speak>
stealing word slots (barge-114: 58 of 66 word-gold ticks). The head
training moved the speech/quiet competition but the model re-picked
WHICH token embodies speech: <speak> (trainable duplex row) now
out-competes words (frozen rows) at word ticks, and the piece
continuation contract only follows WORD tokens — so the model became
lexically silent BY CONSTRUCTION while the scorer still counts <speak>
as speech. The barge "gains" are unverified by ear; the dialog
regression alone fails the probe. Per-lesson pole mixing PERSISTS
through head-row retraining — dialog-116 flipped to a <speak>-chatter
pole, dialog-118 to a mute-ish pole (<speak>-><sil> x287). Five-point
law survives its strongest attack yet: editing the readout at the
source still re-arranges, not separates, the poles.

Implications (owner decision 2026-10-02):
1. GRPO on the 11 rows is contra-indicated as-is: shared linear-readout
   ceiling now empirically demonstrated, AND the scorer counts
   <speak>-only speech -> GRPO would farm exactly this loophole.
2. Any future reward/bar MUST require lexical content (word event with
   pieces) for speech/onset credit — "bar v3: lexical-verified onsets".
3. The pre-declared escalation rung is licensed: LoRA-DAgger (probe
   1b) adds nonlinear trunk capacity; saved hidden states are stale
   under a changed trunk -> on-policy re-roll required (~4 h, 1 GPU).

## D1: STATE DIAGNOSIS — no turn-start signal in the trunk states (2026-10-02, job 7019983, 85s)

**Plain words:** do the model's internal numbers contain the thought
"it is my turn to speak now"? We trained small detectors ("probes")
to find that thought. A good detector scores near 1.0; a useless one
scores 0.5 (a coin flip). **Every detector we built scored coin
flip** — simple or fancy (MLP = a many-layer detector that can find
patterns a simple one misses), within a single lesson, across
lessons, and on lessons it never trained on ("leave-one-lesson-out",
the test that decides whether a fix could work on new conversations).
So the timing sense we kept trying to read out was never in the
model. All earlier reader failures (gate rounds 1–3, linear and head
variants) failed for this one reason, and no reader training can ever
work: there is nothing to read. This also kills the planned LoRA
experiment (a small add-on trained while the big model stays frozen)
— an add-on reads the same numbers, and the numbers hold nothing.

The detailed check, for the record: linear + MLP probes on the SAVED 4,374
free-run states of scale1-pt1 (job 7001927 rollout; gold =
script-clock emission label, train_head_dagger.py:161). Tasks:
T1 word-vs-quiet (gold<n_old), T2 speech-vs-quiet (word|<speak>),
T3 lesson-identity (8-way). Splits: per-lesson 5-fold / pooled
5-fold / leave-one-lesson-out (LOO = the decisive one). No 30B model
loaded; pure torch on one GPU.

| task | probe | per-lesson | pooled | LOO |
|---|---|---|---|---|
| T1 word-vs-quiet | linear | 0.562 | 0.599 | 0.520 |
| T1 word-vs-quiet | MLP    | 0.536 | 0.573 | 0.526 |
| T2 speech-vs-quiet | linear | 0.560 | 0.600 | 0.525 |
| T2 speech-vs-quiet | MLP    | 0.540 | 0.574 | 0.508 |

AUC 0.5 = chance. **EVERY cell is at chance — including per-lesson,
where gate1 once found local signal (and where the pre-registered
"lesson-local basins" theory predicted strong signal).** MLP is NOT
above linear: there is no nonlinear structure either. Per the
pre-registered decision rule (both weak everywhere): no signal in the
states at all -> **readout training is dead in its STRONGEST form —
not "doesn't transfer across lessons" but "nothing to read out".**
This retro-explains all three failed readout attacks with one cause
(gate-MLP rounds 1-3, linear head-DAgger, head-DAgger CE word-acc
0.000): the target was never in the state.

Falsification checks run before accepting the negative:
- Pairing functional? head-DAgger CE TRAINED on these very states to
  acc 0.577 (ctl 0.661) — garbage/misaligned inputs cannot train to
  that. Yes, functional.
- Correct pairing semantics? gold IS the script-clock label at the
  same tick as the free-run state (train_head_dagger.py:161). Yes —
  so the chance AUC says: **the free-run model's speech schedule has
  decoupled from the script clock.** At ticks where the script says
  "speak", the model is usually mid-silence-basin; at its own speech
  ticks the script usually says quiet. The state encodes the model's
  OWN pole, not the script's clock — and no readout of that state can
  recover a clock the model never represented.
- T3 is UNINFORMATIVE (design flaw, noted honestly): the plain-CE
  8-way probe collapsed to predicting the majority lesson (acc 0.000
  vs majority 1.000) on the imbalanced tick distribution. History-
  basin theory is neither confirmed nor denied by T3 — but it no
  longer matters: the main question is answered at T1/T2.

**Consequence: the LoRA science shot (probe 1b) is CONTRA-INDICATED.**
There is no readout signal for LoRA capacity to express; DAgger
labels are script-clock labels, and D1 shows those labels are
unreadable from free-run states. GRPO stays retired (would reinforce
the same poles). The remaining levers are objective-level: what the
labels reward / what the model hears — not another readout of the
same states.

## UGATE: the direct-path composition — same frontier, external mechanism (2026-10-02, job 7019984, ~2h)

**Plain words:** this fix does not touch the model at all. A loudness
meter (voice-activity detection: "is the user talking right now?")
listens to the user's audio; while the meter says "talking", the
model is not allowed to start a new word (words already underway
finish). On the four exam conversations the model never trained on:
interruptions HALVED — the meter really works — but the model then
started its turns too late or missed them (its one-second answer
window shrank from 5/5 hits to 2/5 on dialog-116), and on the
user-heavy "barge" conversations the meter went wrong in a predictable
way: its cutoff is chosen from the loudness levels *of that same
conversation*, and when the user talks most of the time, even the
"quiet" cutoff sits inside speech — so the model was blocked almost
always and barge-113 went completely silent (the audio gallery
confirms: zero model words). Trying a longer block (hold 12 ticks vs
5) changed nothing. **Same trade as every other fix: quieter, but
wrong. A meter that knows only loudness cannot know whose turn it
is.** On the model's practice conversations the results were fine
(dialog-001 4/5, barge-004 4/4) — the meter does not break what
already works; it just cannot create turn-taking that isn't there.

The detailed run, for the record:

The owner-directed "target the problem directly" job: scale1-pt1
(placement solved) + the one untapped INDEPENDENT channel (mic-energy
VAD user-gate, energy from the batch user track — the model's own
voice is never in its ears, so this mask sees exactly what a live mic
sees). --user-gate --gate-hold {5,12} on HELD4 + hold 5 on TRAIN4,
NO min-gap (demo-config policy parity), no training, picks saved.
Panel vs the scale1 closing freerun (job 6971323):

| lesson (HELD) | scale1 base | ugate hold=5 | ugate hold=12 | gate |
|---|---|---|---|---|
| dialog-116 | 3/5, 5/5+-1s, chat 156 | 1/5, 2/5+-1s, chat 94 | 1/5, 2/5+-1s, chat 93 | FAIL exact+-1s |
| dialog-118 | 2/5, 5/5+-1s, chat 184 | 2/5, 3/5+-1s, chat 119 | 2/5, 3/5+-1s, chat 125 | FAIL +-1s |
| barge-113 | 0/3, 1/3+-1s, chat 50 | FULL MUTE | FULL MUTE | FAIL new mute pole |
| barge-114 | FULL MUTE | FULL MUTE | FULL MUTE | PASS (already mute) |

VERDICT: **FAIL, 3/4 held gates at BOTH holds.** Mechanism, from the
gate-block counts (51-102 blocks/dialog, 10/barge) and the per-gold
rows:
1. Chatter suppression WORKS as designed (dialog false-speech ~halved,
   156->94, 184->119). The channel is real and the gate does gate.
2. Onsets pay for it: blocking the word-FIRST pick at a hot-mic tick
   consumes the true onset; gate-hold pushes the retry past the +-1s
   window (dialog-116 5/5->2/5 within +-1s).
3. On dense-user (barge) lessons the within-lesson p25/p95 threshold
   SATURATES: user speech dominates the RMS distribution, so p25
   already sits inside speech -> block nearly everything -> NEW full
   mute on barge-113. Same failure at hold 5 and 12 (hold is not the
   knob; the threshold estimator is).
TRAIN4 (report-only): dialog-001 4/5, barge-004 4/4 (5/5+-1s pre-
gate-era numbers), barge-001 2/4 — matches or beats the gate-MLP
configs on the same lessons; dialog-010 0/5 R 0.01 is its PRE-EXISTING
near-mute pole (R 0.01-0.06 in every scale1-era run), not a gate
regression (only 2 gate blocks fired there).

**The convergent finding (the third and decisive confirmation):**
- training recipes re-arrange the poles (five-point law, 33x scale);
- readout training has nothing to read (D1: chance everywhere);
- an EXogenous, independent gate lands on the SAME frontier — it can
  only re-allocate errors (chatter <-> mute <-> late), because the
  model underneath still does not know WHEN.
The mute<->chatter frontier is therefore a property of the TASK
(labeling: script-clock gold over audio whose labels were never
verified per-clip — 29/120 dialogs survived the filter) + a model
that cannot represent the script clock, NOT of any particular
mechanism. No fourth mechanism on top of this objective will move it.

Standing paths (owner decision, 2026-10-02):
A. Label-quality audit FIRST (cheap, near-free): per-clip ASR of the
   rendered assistant clips vs script (the filter's own lesson-WER
   tooling at clip granularity), then the already-queued dialog
   re-render with per-clip ASR check + FRESH held dialogs. D1's
   decoupling is exactly what inconsistent gold predicts.
B. M2 live loop as the critical path to "audible by ~Oct 14": causal
   VAD + 2b piece-budget scheduling; the demo composite already ran
   end-to-end. Bar v2 on current gold is not the product.
Audio renders of both ugate arms queued (job 7020187) for the ear
check per standing rule.

## DATA GATE — the fail-closed rule is now CODE, with the audit numbers (2026-10-02, later)

**Plain words:** You said: fix the training data, and never again let a
job run on bad data — no proceeding when conversations or timing labels
have failures. That is now enforced in code, and the first measuring
stick (a new audit tool) tells us exactly how bad the old batch was.

**The numbers (new tool `evals/data_quality_audit.py` on scale1's 240
conversations):**
- Only 141/240 passed the OLD factory filter. dense_barge: 117/120
  (fine). dense_dialog: 24/120 (20% — the collapse, now explained).
- Why conversations died: "tail_babble" (the TTS adds words that are
  not in the script) fired 192 times. 39 of the 99 dropped
  conversations had exactly ONE bad clip — the perfect case for
  repair (re-speak that one clip, keep the conversation).
- Worse: of the 141 "passing" conversations, 94 carried individual
  clips above the strict per-clip bar (15% word error). The old gate
  only averaged per channel at 30%, so garbled clips hid inside
  passing lessons. These are clips the model practised on where up to
  80% of the words do not match the script.

**What is now in the code (all committed this session):**
1. **Repair-in-render** (`data/synthetic/render.py --repair 2`): every
   clip is speech-to-text-checked the moment it is spoken, by the SAME
   whisper judge the filter uses. A failing clip is re-spoken (2
   tries, fresh dice); a clip that still fails REJECTS its whole
   conversation — no mix, no manifest row, a REJECTED.json with the
   evidence. Only judged-good clips enter the clip cache, and cached
   clips are re-judged on reuse (old cache entries included).
2. **Timing audit** (`data/synthetic/timing_audit.py`): whisper word
   timestamps vs the answer key's estimated word placement, per
   assistant clip (only assistant words are answer-key labels — user
   speech is the model's INPUT, which is the real audio itself).
   Writes timing_report.jsonl per conversation.
3. **Quality stamp** (`data/synthetic/quality.py`): folds clip
   verdicts + timing verdicts into quality.jsonl — one record per
   conversation, schema dl-quality-v1. Verified means: every clip
   passed AND timing passed.
4. **Startup asserts in every consumer** (8 of them): prepare,
   train_p1, train_gate, train_head_dagger, freerun_diag, probe_p1,
   contract_trace, m2/parity — each now ABORTS with a plain message
   if any conversation it is about to load has no verified record.
   train_p1/train_gate additionally refuse shards with no verified
   conversation (the stale-shard hole, closed). No flags, no
   overrides: fail-closed.

**The old batch under the new gate:** only 127/240 conversations pass
the clip bar (barge 108/120, dialog 19/120) and NONE is fully verified
yet (timing audit has not run on it). The old batch is dead for
training purposes — as intended.

**The fix is already running:** job chain `scale1-quality.slurm`
re-renders the SAME timelines with the judge on (validated clip cache
carries over, so cost is bounded), then chains filter -> timing audit
-> stamp. It STOPS there: re-sharding + retraining wait for the
quality stamp and the owner's go.

## M2 2b BENCH — decisions on time, voicing is the wall (2026-10-03, job 7026771, 4h16m, 5 arms)

**Plain words:** The live loop splits each 80 ms tick into two jobs:
DECIDE (the ear listens, the model picks its next action) and VOICE
(actually emitting its word tokens). We tested the scheduler that
makes DECIDE the only time-critical job, against two maths kernels and
four piece budgets. Result: **deciding fits real time, voicing dense
speech does not — on this checkpoint, with one GPU.**

Numbers (dense_dialog-001, the word-heaviest lesson; all arms parity
0 mismatches — the scheduler never changes what the model says):
- DECIDE-only ticks: p50 **59 ms** with the batched_mm kernel —
  comfortably inside the 80 ms tick. With the eager kernel the same
  ticks take 165 ms: eager is out, batched_mm is the keeper.
- VOICE ticks on dense speech: best arm (4 pieces/tick, batched_mm)
  spends p50 **386 ms per speaking tick** and the backlog of owed words
  grows to ~1300. The script simply demands ~6.6 words per tick; the
  model needs ~80 ms per word. That is ~5x over budget — arithmetic,
  not tuning.
- Quiet lessons (dialog-010, barge-004) already run 94-99% on time.

**What this means for the audible demo (due ~Oct 14):** the loop CAN
decide in real time; the voice channel lags only when the script is
word-dense. Paths: sparser demo turns, speculative piece decode, or
accept lag on dense turns. Fail-closed note: engineering bench only —
no model-quality verdict until the re-rendered verified batch lands.

## SCALE1-Q — first training on FULLY VERIFIED data (2026-10-03, job 7035090, 3h29m)

**Plain words:** We retrained with the exact same recipe as before —
the ONLY change is the data: 153 conversations where every clip
matches its script and every timing label is measured from the real
audio (the first corpus in this project where that is true). The
question was: was bad data the reason the model cannot tell when to
speak? Answer: **bad data was A reason, not THE reason.**

Held-out free-run (4 verified lessons the model never trained on):
- Dialog lessons: still near-mute — the model starts only ~15% of the
  words it should (recall 0.14-0.19) and misses most onsets (0-1 of 5).
- Barge lessons (user talks a lot, assistant interjects briefly):
  the bright spot — "is the assistant speaking" F1 0.53-0.60, word
  accuracy 0.36-0.41, and 2-4 of 4 word starts land within ±1 s
  (2 of 4 exactly right on barge-113).
- The ear-clock skew (52 vs 50 tokens per 4 s) is STILL in the loop —
  flagged by the diagnostic, fix not yet applied. It smears onset
  information over the history, so this readout is a LOWER bound on
  what clean alignment could give.

**What this settles:** data quality is no longer the suspect — the
remaining wall is the model/loop side: ear-clock alignment, then the
gate round on this trunk. Numbers above are the first in this project
measured on answer keys that are actually correct, so from here,
changes move REAL numbers.

## SCALE1-Q2 — THE EAR-CLOCK FIX (2026-10-04, job 7039869, 3h30m)

**Plain words:** The model could not tell when to speak. We fixed the
data — it helped some, not enough. Then we fixed the model's HEARING:
its ear produced 52 sound-snapshots per 4 seconds while the clock
ticks 50 times; the surplus was silently discarded, so everything it
heard drifted 4% late through every conversation — like dubbing that
slowly slips out of sync. After aligning the ear to the clock (same
data, same recipe, ONE change), the model went from near-mute to
answering on time.

Same 4 held-out lessons, before -> after (speech recall / exact word
onsets):
- dense_dialog-116: 0.14 -> **0.93** recall; onsets 0/5 -> **5/5 exact**
- dense_barge-113: 0.48 -> **0.89** recall; onsets 2/4 -> **3/4 exact**
- dense_barge-114: 0.36 -> **0.61** recall; F1 0.53 -> **0.70**
- dense_dialog-114: F1 0.31 -> 0.46; onsets 2/5 exact

**The honest trade:** precision fell — the model now OVER-speaks
(starts words in ~15% of quiet moments it should stay silent in).
Recall up from ~0.15 to ~0.9, precision down to ~0.65 (v3-framed).
That is exactly the chatter-vs-quiet operating point the GATE round
exists to pick — and the gate on this trunk trained while these
numbers were measured. Turn-taking is no longer a hearing problem;
it is a threshold problem, and thresholds are what gates do.

## THE FIRST LISTEN — what the ear caught that the metrics hid (2026-10-05, owner audition)

**Plain words:** The owner listened to the demo and heard two things
the numbers had hidden. (1) The gallery never listed model.wav —
fixed. (2) Deep groaning under the conversation: the model's free-run
word picks are ~93% degenerate single characters ("0", "9", "."), and
the TTS read the digit stream as a drone. This forces an honest
correction to every "word accuracy" number in this file: the
diagnostic's gold-<word> accuracy measures whether the model STARTED
A WORD at the right tick (placement), never WHICH word. Timing is
real and fixed; free-run CONTENT was never trained — phase-1 is
teacher-forced, the words were always handed to the model, and the
Talker that would generate them is not in this trunk. The audible
demo is therefore rebuilt honestly: the MODEL'S turn onsets + the
SCRIPT'S words (evals/build_timing_demo.py) — turns the model missed
are gaps; turns it invented are skipped and reported. What the demo
proves is turn-taking; it does not claim word generation.

## GATE 1 VERDICT — content is closed for this phase (2026-10-11, arm B, job 7249649+7260780)

**Plain words:** The final experiment is in. Training on 434 verified
conversations with corrupted self-history did NOT teach the model to
speak its own words — and it made the dialog lessons mute (recall
0.02-0.03) while one barge lesson babbled more than ever (WER 2.46 vs
0.96 baseline). With decode fixes ruled out earlier and the readout
path closed twice, the conclusion is final: free-run word CONTENT
needs the next architecture phase (Talker wiring or real-audio scale
or RL-after-real-SFT), not another round on this trunk. What the
round DID confirm: placement skills remain trainable and robust
(barge-113: 4/4 exact onsets even in this configuration).
Serving datum: the new NGC-25.10 stack's raw single-token forward is
115ms p50 (vs ~59ms decisions on the old stack) — kernel regression
worth one look before any serving claims.

## DELEGATION ROUNDS — historical negative result; comparison invalidated 2026-10-13 (jobs 7333494/7356783)

**Correction, 2026-10-13:** full job logs and a shell-execution regression
confirm that a blank continuation line dropped `--lessons`, `--init-ckpt`,
and `--out` from BOTH training commands. They ran 1500/800 steps on the
default merged selection, without the intended checkpoint initialization,
and saved to `runs/p1-v1`. The intended 81-vs-30 lesson mix was not the
executed experiment. Consequently the ratio comparison and those candidates'
held-out claims are invalid; poor outputs remain observations, not evidence
that the intended mixtures cannot work. The delivered q2 checkpoint remains
a separate historical reference. See `docs/RECOVERY_PLAN.md`.

**Superseded original interpretation (not a valid mixture verdict):** Two attempts to add delegation/streaming behaviors by
mixing them into training. First (81 lessons): dense turn-taking
broke (146 false <delegate>s on one lesson). Second (diluted 1:6,
800 steps): still broke (dense recall 0.03-0.18). The <delegate>
emission itself fires at the right tick (1/1 in both) — the skill is
learnable, but not mixable into this trunk at any ratio we could
afford. Next phase: staged curriculum or a separated emission head.
DELIVERED MODEL: scale1-q2-pt1 (frozen). Fresh-set final: F1
0.80-0.86, recall up to 0.99, onsets 12/12 within 1s.

## RECOVERY BASELINE — native words and speech work; live contract still fails (2026-10-13)

**Plain words:** The owner reopened the work for the complete duplex/delegation
goal. We tested the untouched Qwen base using its native prompt and full
Thinker/Talker path before choosing another training run. Three verified user
questions, each tested as text and audio, produced six coherent spoken answers.
Whisper recovered the generated text with zero word errors in all six. This
proves a small component path, not generalization, human voice quality, or live
duplex performance. Full-generation time was 12.29–17.94 seconds per answer.

- Job 7377148: first diagnostic stopped on a 24 kHz source clip; no result.
- Job 7377200: native content/speech probe completed in 2m15s; explicit input
  resampling, seed, model revision and full outputs retained.
- Job 7377265: speech audit plus clean-container CUDA check completed in 36s;
  xformers is absent (MoshiRAG environment work remains).
- `evals.live_contract_probe` executed the actual `m2.live.main` scheduler with
  numbered fixture ticks and a fake model/encoder: no features for 49 ticks,
  then consistently 3,920 ms old input. Exit 1 against the 200 ms age budget.
  This is control-flow evidence, not a GPU measurement. The prior 80 ms cadence
  does not establish low end-to-end latency.
- Fixed missing startup quality checks and billion-sample answer-clip padding
  in `m2/live.py`, guarded by `evals.harness.test_live_inputs`. The audio-delay
  failure remains; changing it requires a matching training/inference path.
- Full delegation logs invalidated the claimed mixture comparison; correction
  is recorded directly above its historical verdict.

Next decision: already-duplex native candidates (MiniCPM and MoshiRAG), with
timed output and asynchronous-result tests; then a small checkpoint-reloaded
training pilot. No large training run has started in this continuation.
Current plan: `docs/RECOVERY_PLAN.md`; machine-readable evidence:
`docs/reviews/2026-10-13-recovery-evidence.json`.

## NATIVE DUPLEX + DELAYED CONTEXT — working component, latency still fails (2026-10-13)

**Plain words:** MiniCPM-o 4.5 now runs its own duplex speech on the new
container. Job 7377673 replayed 60 seconds of verified user audio and generated
four answers. Processing was 0.477 s p50 / 1.768 s p95 per one-second chunk,
but input backlog reached 3.095 s. Its bread answer incorrectly blamed drying.

Job 7378025 used the same seed/input plus a fixed reference delivered by a
worker after a ten-second delay. Seven packets completed while the worker
waited, six in speaking mode. The result was ready at 30.000 s and injected
at 30.467 s through a separate text prefill. The generated bread answer then
used the supplied starch-retrogradation mechanism. This is positive evidence
for nonblocking result delivery and use on one case, not learned task dispatch
or actual background research. Cancellation, stale results and fresh tasks
remain untested. Backlog reached 3.913 s (final 3.681 s); processing p50/p95
was 0.526/1.794 s. Both runs still fail the intended live latency requirement.

Earlier jobs 7377413/7377469/7377588 exposed audio-decoder dependency problems.
The working launcher retains container Torch, isolates dependencies and uses a
tested process-local normalized-WAV reader; TorchAudio features are unchanged.
Both complete runs have packet logs, environment/commit records and generated
WAVs in the listening gallery. Human listening remains pending. All eight
recovery jobs together used 514 allocated GPU-seconds (0.143 GPU-h). No broad
training has started. Next: MoshiRAG live baseline and a small reloaded native
Moshi adaptation pilot, with fresh evaluation splits before scale-up.

## NATIVE MOSHI PILOTS — training verified, content not improved (2026-10-13)

Jobs 7380342 and 7380505 pass strict base loading, finite/nonzero adapter
gradients, unchanged frozen-weight hashes, exact saved-adapter hashes and
fresh-process canonical float32 logit comparison (0.0 difference). The earlier
7379340 BF16 comparison failed; diagnostic 7379832 found exact weights but
its numerical discrepancy is not fully explained. Keep that failed verdict.

Job 7380505 trained 64 steps over 16 reused conversations, with four diagnostic
examples. Loss 1.9321053 versus the four-step pilot's 2.20383549 is not proof
of generalization or useful spoken content. Generated answers remain weak.
Paced 80 ms frame processing p50/p95 was 0.0220/0.0236 s; this is not response
latency. Human listening and fresh evaluation remain open. Latest sampling
seed placement is corrected after warmup; historical outputs predate that fix.

MoshiRAG kernel job 7379661 passes on GH200. Four actual-upstream reference
lifecycle regressions now pass; full encoder/reference use awaits the gated
Meta tokenizer. All jobs are terminal. Combined recovery allocation: 2,590
GPU-seconds (0.72 GPU-h). No broad training or production selection.

## ARC + NATIVE REQUESTS — encoder/application pass, late-answer behavior fails (2026-10-13)

Owner-approved Meta tokenizer access verified by actual pinned-file download.
ARC probe 7386266 passes strict inference-weight coverage, finite nonempty
embeddings and actual HTTP delivery (51–64 ms). Only two source-unused released
checkpoint tensors are omitted explicitly; no inference weights are missing.

Correction: speech probes 7386447/7387021/7387196/7387230/7387611 queued
references but omitted the upstream runner's explicit per-frame application.
Their negative reference-use verdicts are invalid. Seeded pre-worker variation
remains real; longer warmup, fresh processes and eager execution did not prove
identical histories. Eager input-code and traced RNG hashes matched, but output
tokens differed. Cause unestablished. Eager backlog 39–49 s fails live use.

The omission is fixed with two actual-upstream CPU regressions and framewise
application/clearing checks. Corrected 7387777 applied all 33 reference rows.
Four native RAG requests launched ordered fixed-fact workers; frame processing
p50/p95 16.5/18.6 ms, max input backlog 78 ms. Ten-second bread result arrived
45.271 s and was queued at 45.280 s, then applied in frame 565, after a wrong bread explanation; no correction
followed before the next request. Long-delay behavior fails this case. No
actual research, semantic routing or general learned-delegation claim.

Audio audit 7387786 confirms core answer words, but ASR invents repetitions at
the truncated final answer beyond file length; full WER is not a valid quality
verdict. Original WAVs are in the listening gallery; human review remains open.
Total recovery allocation: 3,936 GPU-seconds / 1.093 GPU-hours, including failed
runs. All jobs terminal. Next: fresh reference-fact holdouts, real worker
integration, and causal timed-reference adaptation; no broad training selected.
# 2026-10-14 — Actual GPT-Live reference and renewed comparison

Plain words: GPT-Live access and an audible reference replay now work with the
existing cluster credentials. CPU access attempt 7400572 failed on a malformed
working-directory line before any API request; corrected 7400621 completed.
Paced replay 7400991 completed with 77 seconds usage, 752 audio packets, no API
errors and no delegation on four general-knowledge questions. Received text
contains correct starch-retrogradation reasoning. Speech quality still needs
human audition; these are reused development inputs, not independent holdouts.
No backend call occurred, so the context-based worker code is unvalidated by
this run. The replay is a voice reference, not a matched-worker comparison.

Artifacts: `runs/gpt-live-replay-7400991/`, full retrieved CPU logs in
`runs/recovery-20261014/`, and gallery
http://localhost:8766/batch/gpt-live-replay-7400991 .

PersonaPlex refresh 7400573 is capped at one GPU for 15 minutes, uses pinned
source 3428dfd95309a7f3c84fd93259ded0f810d1ff91 and weights
fdaf4090a61cb315c138a1faee287ffd6c716309, with the same approved 60-second user
stem plus ten seconds of silence. It remains pending for resources at this
entry. No result or GPU-hours are attributed to an unallocated queued job.

Verdict: actual reference recording available; backbone choice, delegation
selection and timed-reference training remain open. See `docs/RECOVERY_PLAN.md`.

## 2026-10-14 — Causal reference training seam and parallel probes

Plain words: the owner confirmed seven days remain and authorized needed compute.
Whole-sequence training can now receive reference rows only after their completion
time. Native CPU job 7401774 passes three tests: comparison with one-frame native
execution, no future-reference leakage, and backward/checkpoint-reload checks.
These use a tiny model constructed from real pinned source, not released weights.
Initial 7401729 failed on an unsupported one-codebook test fixture; fixed to two.
Strengthened reference-effect assertions also pass in 7401911: reference rows
change later predictions without changing earlier predictions.

Full released-weight backward proof 7401863 is queued behind the passing test,
using the real ARC HTTP encoder and verified development data. It performs no
optimizer update and does not produce a quality candidate. Independent native
Qwen text-policy diagnostic 7401909 and PersonaPlex audition 7400573 are also
queued. GPU capacity is presently unavailable; pending jobs have no allocated
GPU-hours. Do not report any of these queued GPU probes as passed.

Verdict: causal conditioning passes native CPU engineering checks; full-weight
training, real delegation and fresh audible improvement remain open. Complete
CPU logs are retrieved under `runs/recovery-20261014/`.

## 2026-10-14 — Interactive scheduling verified; full-weight prefix failure

Plain words: the official interactive reservation starts short development work
without the normal queue delay. Job 7402644 started on a GH200 at the first
follow-up check. Original held 7401863 was cancelled to avoid duplicate work.
The native ARC HTTP encoder passed again, but whole-sequence speech training
failed before backward: a first-speaker embedding of dimension 4096 was being
concatenated with 128 token-code time steps. The CPU reference tests remain
valid; they did not exercise this released prepend conditioner. Prefix-aware
training/inference parity remains open. Do not omit the prefix to declare a pass.

PersonaPlex 7400573 failed after 27 allocated GPU-seconds because the runtime
CUDA linker library was absent from Triton's search path. The existing job-local
linker helper is now used; repaired interactive probe 7402673 completed in
67 GPU-seconds with 70 seconds of generated audio. Its transcript gives a wrong
moisture-loss explanation of bread staling; audible quality is not yet graded.
Qwen replacement 7402713 was accepted in the interactive reservation and its
held workq duplicate 7401909 was cancelled. Retrieve full logs before analysis;
7402644 and 7400573 logs are local in runs/recovery-20261014/.

Verdict: interactive scheduling works and accelerates debugging. No new quality
candidate or full-weight training pass is established by these failed probes.

PersonaPlex artifacts are local at `runs/personaplex-recovery-7402673/` and
available in the gallery; both input/output players were verified. Newly
terminal GPU allocations 7400573/7402644/7402673 add 153 GPU-seconds, taking
recorded recovery consumption to 4089 seconds / 1.135833 GPU-hours before any
Qwen replacement allocation. Interactive billing is separate; this is not the
remaining project balance.

Qwen 7402713 started in the same second as submission, then was killed by the
host-memory cgroup during checkpoint loading (50 GPU-seconds; no routing cases
completed). The launcher's host-memory request is increased from 64 to 110 GiB
for a one-GPU retry. Total recorded terminal recovery GPU allocation is now
4139 seconds / 1.149722 hours, excluding the upcoming retry and billing premium.

One-GPU Qwen retry 7402749 submitted via the interactive reservation with
110 GiB host RAM. Interactive starts measured from Slurm Submit/Start timestamps
were 1 second (7402644), 2 seconds (7402673), and 0 seconds (7402713).

## 2026-10-14 — Native conditioned optimizer and exact fresh reload

Prefix-aware tiny-model regression 7402827 reproduced the released prepend
failure. Repaired whole-sequence embedding construction retains the speaker
prefix and native delays/target masks. Jobs 7402863, 7406083 and 7406692 pass
four, five and six progressively strengthened native CPU tests respectively.
Full released-weight backward 7402880 completes using the actual ARC HTTP
reference encoder; no optimizer or quality candidate was produced by that job.

Native LoRA rank-8 optimizer pilot 7406181 performs four finite updates on a
reused ten-second engineering sample and saves the adapters/optimizer. It then
fails at reload because the upstream convenience loader casts two float32
speaker-conditioner adapters and the frozen conditioner to bfloat16. The exact
loader preserves each tensor dtype. Fresh process 7407067 passes with maximum
text/audio logit error 0.0 on the same saved checkpoint. Frozen weights stayed
unchanged. Adapter SHA256:
`188bb071747f0839f37342f023d5b8e65e038ec84b789f2c41e9a46d8ae0efd0`.
This establishes train/save/reload correctness, not speech quality.

Qwen 7402749 completed but used incorrect direct-generation stopping behavior;
18/20 outputs violated strict JSON. Correct native wrapper retry 7402864 gives
11/20 manually labelled routing cases, 8 invalid outputs and 6 missed required
delegations. This prompted text diagnostic is not learned duplex delegation.

Fresh TTS generation 7406244 failed after a request timeout. Resume 7407148
completes with 181 reused unverified clips and one newly generated clip: 40
scenes, 182 clips, 733.76 seconds of canvas. Voice/model provenance is recorded.
Sealed folds: 24 train, 8 validation, 8 test; assistant voice is shared, script
families and user voices disjoint. Verification 7407250 submitted on workq;
no data or quality-training pass is attributed to its submission.

Verdict: native conditioned adapter serialization repaired; fresh data approval,
generated-speech improvement and real semantic worker dispatch remain open.
Terminal GPU jobs 7402749/7402864/7402880/7406181/7407067 add 679 GPU-seconds
to the previously recorded 4139: 4818 seconds / 1.338333 allocated GPU-hours.
CPU/TTS jobs and interactive billing premiums are separate from this total.

## 2026-10-14 — Fresh native training and actual worker comparison

Immutable repaired TTS source 7407535 has 38/40 strictly approved scenes in
verification 7407554 (23 train/8 validation/7 test). All four behaviors remain
in every fold. Original workq verification 7407250 and held original downstream
jobs 7407385/7407396 were cancelled after replacements; no rejected data trained.
Frozen native Mimi/interleaver/ARC cache 7408223 preserves all measured tokens.

First 32-update training 7408245 fails before its first optimizer step because
no_compile ended before checkpointed backward. Fixed 7408323 completes 32 steps,
674 changed adapters, unchanged frozen weights and fresh-process text/audio
logit errors 0.0. SHA256 d628e63b402b08ef19add3eb13ff6865dae179d5f86e950e7699f0e58cb32708.
Weighted workq 7408653 completes 128 steps (225 allocated GPU-seconds), delegation
weight 10/result-word weight 3, exact fresh reload 0.0. Adapter SHA256
9e3cbd1fd24c4bf1bf29a9b7de075018cfa30423c2d73955cd1123796f2ebe98.
No generated quality claim follows from teacher-forced loss improvements.

Actual semantic worker 7407536 passes 8/8 backend development cases with three
real read-only record calls. Plan-versus-post-tool response handling and required
execution guard repaired earlier failures; this is not native speech judgment.
Released-base voice 7408188 starts the private worker at 26.10 s; native 32-update
7408416 starts at 12.74 s. Both apply all returned ARC rows but omit yellow circle.
Base claims completion early. One replay cannot establish a causal learned timing
gain. Whisper audit 7408487 confirms generated omissions, valid word/segment
timings, WER 0/.0189 respectively. GPT-Live actual comparison 7408224 calls the
record tool, returns yellow circle in audible speech, and answers science;
Whisper WER .0256. None is general product qualification or a human voice score.

Raw GPT-Live teacher CPU 7408766 fails after one completed recording on a floating
sample-length comparison. Integer sample-count regression and safe unverified
resume preserve the completed record, archive incomplete attempts and verify all
hashes. 7409126 completes twelve TRAIN-only scenes (8 private delegations, 4 simple
without delegation). Native conversion/grid approval still required. Held-out
audio is excluded. Actual API retry usage is separate from allocated GPU hours.

Captured reference tests 7408865 fail on unexported DL, 7409127 on divergent paired
histories, and 7410405 on pre-injection slot differences .125 despite fixed approved
histories (post-reference difference 5.625). These are INCONCLUSIVE, not a proven
reference influence. Single-slot repeated controls are the next diagnostic.

Next jobs: 7410566 weighted-128 generated voice; 7410567 matched weighted-32 workq
comparison. Independent teacher speech/fact audit is committed next. Human voice
preference pending at the focused local comparison page; no product model selected.

## 2026-10-14 — Captured reference proof and first spoken result

Single-slot 7410621 remains inconclusive under the normal attention backend:
control-repeat error .1875, reference-prefix error .125. Deterministic math SDPA
7410675 passes with both errors exactly 0.0 and post-reference difference 5.6953125
(actual frozen ARC injected at 6.32 s). This isolates reference transport through
captured temporal inference; it is not a free-running task or latency result.
Production attention remains unchanged.

Voice 7410566 fails before generation because its launch argument named a
directory rather than lora.safetensors. Broken pending cancellation/simple jobs
7410623/7410624 were cancelled before compute, replaced by 7410642/7410643.
Correct private replay 7410641 completes, requests actual record reads at 5.14
and 12.76 s, applies ten ARC rows and speaks yellow circle. It also guesses red
early and repeats answers. Verdict FAIL. ASR audit 7410741 confirms actual speech
with WER .01695 and valid timestamps. Full recordings/receipts are local.

Matched weighted-32 workq 7410567 was held, replaced by bounded interactive
7410830, then cancelled; no duplicate GPU work. GPT-Live teacher GPU audit
7410622 was likewise replaced by explicit CPU 7410766 with the same judge and
threshold. Train-only measured conversion/composition is committed, with no
word interpolation, 200 ms grid bar and unchanged sealed validation/test rows.
Independent audio/fact approval and native codec/marker checks still gate use.

## 2026-10-14 — Approved natural teachers, exclusive GPU fix, behavior failures

CPU speech/fact audit 7410766 accepts 10/12 raw GPT-Live TRAIN recordings (six
private/four simple); two exact measured-word mismatches stay excluded. Derived
conversion/composition 7410969 yields 33 train/8 validation/7 test. Original
held-out rows are unchanged. Recorded input is verified against approved user
stimulus plus silence; current approval checks re-run at ingestion startup.

Teacher encode 7411223 fails before codec loading because a second process cannot
open the GPU. Its dependent training 7411224 is cancelled without allocation.
Single-context 7411513 completes on `Exclusive_Process` while using the actual
HTTP encoder, adds ten native teacher caches and reuses 38 verified base caches.
Actual first-subword timing is checked in addition to the measured word grid;
all ten pass. CPU structural audit of all 38 base caches passes, max offset 80 ms.

Teacher-mixture training 7411514 completes 128 updates: 674 changed adapters,
unchanged frozen weights, fresh text/audio logit errors 0.0. SHA256
104bcd641351e03f8bf7da6f00477bf63c9d8a4f369a75c5d1b9bd7247753093.
Generated 7411691 still omits the private value despite three real record reads.
The result-utterance weight covers the entire merged teacher assistant stream;
`result_word_nll` for that source is not private-phrase accuracy. Narrow the
supervision scope before enlarging this objective.

Matched weighted-32 7410830 saves/reloads exactly (SHA
7f4e3831ae83f00d5b8074a7da3c95466dd55c0d871b5a4055cabf4a9c42f294).
Generated 7411280 invents a white marker and a book. Verdict FAIL.
Simple original weighted-128 replay 7410643 answers salt without worker calls,
but cancellation 7411020 guesses location and speaks red star after cancellation.
Its actual backend cancels request-325 (its own cancellation dispatch), leaving
older lookups live. Captured self-cancel test is red before fix, green after.
Existing-task selection excludes current dispatch; atomic selected-ID batches
preserve unrelated work and reject invalid targets without partial changes.
Actual API cancellation component 7411442 passes. Native 7411443 never dispatches
cancellation, so that replay does not prove end-to-end cancellation repair.

Scheduling race: intended pending replacement 7410642 had started before the
move; it was cancelled after 104 allocated GPU-seconds, then rerun as 7411020.
Count that allocation. Do not describe it as a zero-compute pending replacement.
Check state both before and after holding a job; do not cancel a running job
merely to move its queue.

Chronological user turns and generated assistant chunks replace flattened user
history. Actual ready-phase metadata helps the worker avoid repeated old reads;
future words and scripted assistant answers remain excluded. Context is still
an oracle, not real ASR. Backend regression 7411885 passes 8/8 policy and 8/8 tool
contracts with three actual reads. Native speech judgment remains unqualified.
Job-local ephemeral HTTP endpoints and single-process voice allow parallel
replays, including 7411886/7411887 on nid010336, with no cross-job port collision.

Teacher checkpoint matrix: 7411889 private uses yellow circle after one actual
read but adds unsupported claims; 7411886 simple requests work twice and gives
an unclear answer; 7411887 skips clarification and invents books; 7411888 misses
cancellation and makes premature private/completion claims. All fail overall.
Compute p95 22.36–24.35 ms/80 ms, max input backlog 33.4 ms. This is software
throughput, not microphone or general task acceptance. Audit 7412064 retains all
recordings; private aggregate WER is invalid because Whisper hallucinates a tail
outside real time bounds. Keep that failure explicit. Early bounded words place
yellow circle at 10.48–11.08 s after actual ready 8.55 s; no whole-task pass follows.

27 current local context/cancellation/alignment regressions pass. Inference-only
text-temperature .2 probes 7412588 private and 7412590 cancel were submitted;
results are unread after Clifton expiry at 19:30:50 UTC. Renewal launched; owner's
identity-provider sign-in is needed. Do not duplicate unknown-state jobs.

Complete accounting snapshot through 7412064: 80 terminal GPU allocations,
7692 GPU-seconds / 2.136667 GPU-hours, including failures/cancellations. This is
not remaining project balance and excludes the last two probes, billing premium
and CPU/API costs. Local backups verify all 48 code tensors and 48 audio/sidecars;
parent metadata archive is incomplete after expiry. Latest code through 61d667a
was pushed and pulled; this ledger/handoff update requires cluster pull after
login. No candidate is delivered as a qualified model. Voice preference pending
at http://localhost:8766/audio/voice-comparison/index.html .

## 2026-10-14 — owner rejects audible artifacts; packet/codec controls and native alternative

**Plain words:** the owner prefers PersonaPlex over MoshiRAG, but reports clicks,
crackling and broken speech in both. Neither voice is accepted. We separated
recording gaps from generated sound, compared the same codec tokens on CPU/GPU,
made additional real auditions, fixed an overly broad training weight, and
prepared a strictly loaded alternative model. These checks do not deliver a
qualified voice or delegation model. Authentication is restored and verified.

### Completed evidence

| Job / artifact | What happened | Meaning |
|---|---|---|
| 7412588 / 7412590 | Both temperature-.2 replays complete; all logs/audio retrieved after renewal | No duplicate submissions |
| 7414528 | Speech audit: private WER 0 but says only yellow; cancel WER .2174 with valid bounds | Both tasks fail; lower text temperature does not repair them |
| 7414519 | First codec reconstruction fails after 58 GPU-seconds on the known linker-path issue | Applied the existing job-local CUDA linker repair |
| 7414595 | Codec reconstruction and NATF1/NATM1 auditions finish; custom prompt fails on missing pyloudnorm | Full allocation 233 GPU-seconds, including the failed condition; do not discard completed controls |
| 7414865 | Custom prompt fails after 87 GPU-seconds on pyloudnorm's missing future dependency | Isolated overlay now supplies both and verifies imports before loading models |
| 7415052 | Uncompiled GPU decoder control and custom PersonaPlex prompt both finish, 111 GPU-seconds | Real custom voice recording exists; factual answers remain wrong |
| 7415051 | CPU decodes the same 240,000 samples / eight-codebook token stream without compilation | RMS difference from compiled GPU .00003106, maximum .002305; no perceptual pass implied |
| 7415052 decoder | Same GPU tokens without compilation | RMS difference .00001817, maximum .0003555; large compilation divergence is not supported by this control |
| local moshi-playout-diagnostic-20261014 | Exact old capture reproduced from ledger; two inserted-silence gaps total 122.54 ms | A causal 160 ms startup buffer removes both while preserving every model sample |
| 7415050 | Fresh actual GPT-Live: 77 API seconds, 753 audio packets, zero errors/delegations | Transcript answers all four science questions correctly; still one development replay |
| 7415050 playout | Original capture 76 underruns / 1.312 s; 160 ms buffer leaves 66 / 1.152 s | Startup buffering alone does not fix this transport stream; raw PCM exported separately for voice-only audition |
| 7415165 | Native MiniCPM with synthetic-marin reference; 97 GPU-seconds | Only the first science answer; p50/p95 .452/1.836 s per second of input, maximum backlog 6.229 s; fails content/latency |
| 7414927 | CPU checked-phrase supervision audit passes all 23 eligible TRAIN/validation scenes | Sealed test labels excluded; six natural teachers weight 2–3 actual phrase subwords instead of 34–55 whole-stream tokens |
| 7414609 | CPU downloads pinned Lychee-FD model/vocoder/source and hashes all files | Evaluation assets, not a model qualification |
| 7414719 / 7414866 | Isolated CPU environment checks fail on optional TorchAO API, then missing datasets | No dependent GPU jobs run; no global Torch/library replacement |
| 7415053 | Isolated environment and actual native-source imports pass | BF16 only; optional unused quantization discovery explicitly disabled |
| 7415211 | CPU preflight matches every one of 982 native checkpoint names/shapes | No weights loaded or GPU used; no half-duplex class silently substituted |
| 7415227 | Native model loads with zero missing/unexpected/mismatched weights, but emits no events; 92 GPU-seconds | Wrapper used wrong audio/control alignment; not an audible model comparison |
| 7415324 | Real native preprocessing: old setting yields five slots/400 ms and fails; published alignment yields ten and passes | The wrapper must match the native 25 Hz control clock |

Corrected native Lychee audition 7415325 depends on the real clock check and
retains text/speech/control token traces. Result pending at this entry; do not
claim it sounds better. Pins: source 7bb1068c59b578e1f8e65fb762cd26708c14cc6d,
model ac06087bef70b33d4dfebf63a3eadadb4e024f33, separate vocoder
e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36. The explicit compatibility bridge uses
the released training implementation because HF's packaged automatic loader
names a simplex class and the serving runtime imports an absent module. Strict
name/shape/loaded-weight checks prevent dropping the extra branches.

### Scope, scheduling and records

PersonaPlex's published offline exporter already concatenates generated frames
continuously, so the demonstrated playout gaps do not explain its reported
artifacts. The gallery returns exact WAV bytes. PersonaPlex and GPT-Live are
mono 24 kHz, finite, without clipped samples. Do not infer perceptual quality
from these signal checks or ASR agreement. The original GPT-Live excerpt used
for codec/voice prompts is arrival-aligned and may contain transport gaps; it
is not a pristine codec ceiling. The fresh raw PCM stream is an explicitly
untimed diagnostic, not evidence of live latency. Future GPT-Live/native recovery
captures keep original timing WAVs plus buffered/raw versions and exact receipts.

Temperature-.2 cancellation does issue a real cancel request at 22.58 s after
the user's request ends at 19.42 s. The actual lookup was already queued at
22.48 s; application cancellation completes later. This exercises the repaired
application but still fails timely native cancellation and spoken content.

The checked-phrase loss regression fails on the historical whole-stream mask and
passes after the fix, including guards against premature/missing/reordered
subwords. New default scope is measured-phrase; last-utterance explicitly
reproduces old experiments. No optimizer run with this changed scope is claimed.
Relevant local suite: 36 tests pass, including causal context, cancellation,
source/timing alignment, reference WAV/sample counts and playout stalls.

Pending-only or dependency cancellations 7414520, 7414537, 7414608, 7414721,
7414812, 7414867 and 7415054 consume zero allocated GPU time. 7414520 was an audit
payload accidentally submitted without its intended GPU/time flags; it was held,
rechecked as pending, cancelled, and replaced by bounded 7414528. Moving pending
7414812 to the reservation hit QOSMaxSubmitJobPerUserLimit; no QoS bypass was used.
7415165 was submitted after the slot freed. Never cancel a running job to move
queues. Interactive work here is finite diagnosis/audition, not bulk training.

Physical allocation snapshot through 7415227: 89 terminal GPU jobs, 8,652 GPU-seconds
/ 2.403333 hours, including failures/cancellations. Excludes CPU allocation billing,
interactive premiums and pending corrected replay; not remaining project credits.
Full sacct export and computed receipt are local under recovery-20261014.
All 48 mixed code/audio payloads remain hash-matched locally; interrupted parent
metadata retrieval is complete and all three parent ingestion hashes match.

Focused listening: http://localhost:8766/audio/voice-diagnostics/index.html .
The top players compare raw GPT-Live voice, original PersonaPlex and the actual
custom voice condition; technical controls are collapsed below. Canonical
continuation and exact paths: docs/RECOVERY_PLAN.md and HANDOFF.md.


## 2026-10-14 — Owner accepts custom PersonaPlex and Lychee voice quality

**Plain words:** the custom synthetic-reference PersonaPlex (7415052) and corrected
Lychee-FD (7415325) both sound like GPT-Live to the owner. Original NATF2 sounds
muffled/radio-like and remains rejected. This closes the listening choice for
this stage; intelligence, real-time speed and delegation are still unqualified.
Lychee strictly loads all 982 tensors and speaks four science answers, with the
bread-staling answer wrong. Offline generation plus vocoder: 229.47 s / 70 s input.
Keep the accepted synthetic reference fixed and test causal background-result
integration and paced performance before selecting the larger fine-tuning run.

Revised measured-phrase training 7415494 completes 128 updates in 3m37s. Fresh
process reload text/audio max errors are 0.0; adapter SHA
0263c20d64a7649c9efd43a659e4fb31503e36c553071128531f78738577dc9f.
Dependent development jobs: private 7415517, simple 7415519, clarify 7415521,
cancel 7415522. Completion or exact reload alone is not a behavior pass.


### 15 October continuation — accepted voice paced; bridge engineering proof

**Plain words:** accepted custom PersonaPlex runs at 30.47/32.46 ms median/95th
percentile per 80 ms input frame (7415855). Maximum input backlog is 161.12 ms;
a 160 ms startup buffer leaves one 50.71 ms playback gap. This is promising
compute speed, not clean microphone acceptance. Native science answers still fail.
The replay changes no native model, prompt, sampling or generated-frame export;
additional WAVs retain real packet availability and an untimed PCM diagnostic.

Revised MoshiRAG objective pilot 7415494 still fails all four development tasks:
7415517 gives only yellow, invents/repeats room claims and rereads the record;
7415519 answers salt dissolution but dispatches two unnecessary workers;
7415521 skips clarification and invents marker facts; 7415522 never cancels.
Independent audit 7415856 has valid measured word bounds and respective WER
0.03125 / 0.02326 / 0.06667 / 0.0. Intelligibility does not rescue task failures.
Do not scale this recipe based on lower loss or exact reload.

PersonaPlex reference proof 7415929 passes on actual native single-slot generation
with CUDA graphs and identical teacher-forced histories: repeat, zero-initialized
projection and causal prefix errors all 0.0; manually enabled positive projection
changes post-result logits by 1.25. The rank-64 bridge maps frozen ARC rows into
PersonaPlex's temporal input. This proves a compatible causal data path, not learned
result use, speech quality, or native delegation. It assumes no native RAG token.
That first proof omits the voice prompt; a further run checks the exact accepted
synthetic voice and system prefix before any larger fine-tuning.


**Plain words:** native bridge proof 7427733 repeats the check with the exact
accepted synthetic voice and system prompt. Repeat, zero-projection and causal
prefix errors remain 0.0; the deliberately enabled positive projection changes
post-result logits by 1.1875. Native token 4 is `<0x00>`, not a tool command.
PersonaPlex and the approved cache have identical tokenizer and Mimi hashes.
The next bounded engineering pilot trains only the rank-64 reference bridge on
one independently verified TRAIN result phrase, with the actual native delayed
voice/system prefix captured from LMGen. It replaces old RAG markers with native
padding, checks streaming/whole-sequence equivalence before gradients, freezes
and hashes the speech weights, and requires exact fresh-process bridge reload.
No routing, generalization or free-running speech acceptance follows from that pilot.


## 2026-10-15 — Native accepted-voice bridge can train and reload

**Plain words:** the reference bridge now receives actual gradients while the
PersonaPlex speech model remains frozen. Pilot 7428894 uses one strictly approved
TRAIN phrase, the accepted synthetic voice prefix as context only, and no held-out
training. Native delayed inputs/targets are captured directly from LMGen; old RAG
control markers are replaced by native padding. Streaming versus full-sequence
training predictions differ by at most 0.0001847 in float32. Zero bridge and
reference-free prefix errors are exactly 0.0. Frozen speech hashes match. Eight
steps move phrase loss 9.2622 to 5.2089; no free-running quality claim follows.
Fresh-process reload is exact (0.0), bridge SHA
76db170d7f15085ba1241d127f67e48a5b92f4b973f183dc3ab1901b7077f919.

Lychee incremental benchmark 7428971 supplies only arrived 400 ms user chunks,
retains all native streams/events, and uses the published HF incremental runtime.
It measures speech-LM tokens; separate subsequent vocoding stays offline. Results
pending. QOSMaxSubmitJobPerUserLimit rejected additional interactive submissions
without allocating GPUs; a normal workq submission accepts actual-worker probe
7429089. It uses the trained bridge and a real API record read/frozen ARC encoder,
with an explicitly scheduled diagnostic request, not learned native routing.


## 2026-10-15 — Real worker transport succeeds; spoken result use fails

**Plain words:** 7429089 stops before generation because a new asset check wrongly
requires an ARC config.json that its manifest never declared. The correction
iterates declared files, still requires weights and verifies hashes; path preflight
passes. Corrected 7429163 executes an actual yellow-circle lookup, finishes frozen
ARC encoding at 10.294 s, and applies all three rows after completion. Its text
does not give the checked value. No native routing was claimed: dispatch is
scheduled after measured input. Thus the eight-step bridge remains an engineering
pilot, with no useful free-running result-use pass. p95 compute 31.65 ms/80 ms;
max backlog 117.19 ms and one 7.75 ms buffered gap. Native audio audit pending.

Lychee native incremental 7428971 takes 243.26 s / 70 s with 174.63 s max backlog
and p95 1.781 s/400 ms. Bounded encoder memo 7429187 avoids 1,279 encodes but still
takes 214.04 s (143.96 s backlog). All three native token streams differ between
runs; no unchanged-behavior or pure-encoder parity established. Do not adopt it.

Control-teacher 7429356 collects all seven approved TRAIN clarification/cancellation
stimuli. Independent audit 7429624 queued; result unread. New gates demand actual
receipts before cancellation acknowledgements and clarification before identity.
Two additional world-lineage tests and three control-timing/receipt tests pass,
alongside the historical teacher checks. Native history proof 7429698 is submitted
on another TRAIN scene; its result is unread at the authentication boundary.

Paired counterfactual teacher code is prepared/committed, **not submitted**: 32 raw
sessions, four different private values per identical approved TRAIN question, two
concurrent sessions. Provenance, immutable source question, declared world hash,
actual record receipt and strict measured words all remain required. This targets
result dependence rather than project-to-value memorization. API budget is already
authorized. Clifton expires at 10:29 Berlin; SSH public-key rejection blocks the
cluster pull/submission and retrieval. Renewal is started with identity-provider
page open; owner must sign in. No new permissions/configuration changes requested.

**Plain words — local verification:** 41 related tests pass, including the new
control-teacher causality and counterfactual-world lineage guards. No local tests
are claimed as native voice, intelligence or complete-task acceptance.


## 2026-10-15 — Renewed access; paired teachers running; control targets rejected

**Plain words:** SSH renewal is verified at 11:43 Berlin. Cluster pulls committed
code; new paired-private teacher job 7434240 runs 32 TRAIN-only sessions. Independent
strict speech/fact/receipt audit 7434249 depends on its successful completion.
Native history 7429698 passes: prepared inputs on TRAIN-001 exactly match actual
LMGen, using the accepted 162-frame prefix captured on TRAIN-000. No held-out labels
train. This permits efficient multi-example preparation, not model qualification.
Control audit 7429624 rejects all seven raw teachers; none enters training. Inspect
complete events for missing clarification, premature/wrong lookups and cancellation
without receipts followed by private disclosure before choosing the repair.


## 2026-10-15 — Joint PersonaPlex proof catches native training seams

**Plain words:** joint proof 7434416 makes no optimizer updates. Delayed targets
match exactly and text-logit error is 0.0000763, but audio error is 0.71628.
Zero adapters preserve both outputs exactly, yet some adapter gradients are
absent. The proof correctly fails; do not launch training from this result.
Source inspection finds native gated feed-forward code reads raw weights,
bypassing child Linear forwards. Its explicit adapter-aware forward retains
the frozen parameter names and needs native zero-output and gradient checks.
The pinned streaming depth cache excludes its oldest slot when full, whereas
the unstreamed helper includes it. A batched-only compatibility mask is being
tested against every native codebook; vendor source and streaming inference
remain unchanged. The original discrepancy is retained in the new report.
Paired teacher 7434240 completes all 32 raw recordings; audit 7434249 is running.

**Plain words — follow-up 7434638:** stops before gradient work because the
compatibility check assumed depth context 8 from loader options. Native LMModel
explicitly overrides it to None, giving a 16-slot ring. This is a failed source
assumption, not a numerical pass. The next comparison uses the actual 16-slot
boundary (batched context 15), still preserving unmodified per-codebook errors.


## 2026-10-15 — Native joint proof passes; thirteen paired teachers converted

**Plain words:** 7434702 proves exact native delayed inputs/targets, text error
0.0000763 and audio error 0.000124. The original unmodified batch discrepancy
is confined to depth codebook 16 (0.71628); batched context 15 matches the actual
16-slot native streaming cache, without vendor or streaming inference changes.
The explicit gated-feed-forward path gives every intended adapter a finite,
nonzero output gradient. Both zero-adapter errors are exactly 0.0; no optimizer
updates occur in this proof. 3,500,032 language-adapter parameters are checked.

Paired audit 7434696 independently checks all 32; 13 pass, 19 fail. Silent audio
is explicitly failed, never supplied invented words. Regression checks exercise
the real audit loop on silent then audible clips and confirm silence rejection.
Conversion 7434749 stamps all 13 with unchanged WER .08 and measured word clocks;
native encoding remains required. A new trainer accepts multiple independently
validated caches, forbids extra held-out folds/duplicate IDs, and keeps explicit
source hashes. Its separate action head preserves approved control clocks;
two tests reject missing/misaligned markers and distinguish cancellation.
Eight-update native pilot is prepared for engineering validation and fresh-process
text/audio/action reload, not larger training or model qualification yet.

**Plain words — pilot and paired encoding:** native pilot 7434925 performs four
updates then fails on a later native undefined acoustic target; no final model
or reload is produced. The native interleaver uses -1 for undefined padding.
Loss now excludes only native undefined/initial tokens, records excluded counts
per codebook and rejects any other out-of-range value. The original targets and
WAVs are unchanged. The fresh retry must complete and reload before inference.
Paired encoding **7434926** completes all 13 with native first-word errors at most
100ms, below the unchanged 200ms bar. These may now join the 33 approved TRAIN
scenes as a separately hashed cache, with old validation/test folds unchanged.

**Plain words — control contract repair prepared:** raw control 7429356/7429624
remains rejected. Its events contain acknowledgements but no client cancellation
requests. New optional, versioned client-control-v1 instructions explicitly require
requesting cancellation and awaiting its receipt, and asking which project before
private dispatch. It applies only to control TRAIN teachers and is pinned in each
recording's environment; old collection prompts remain unchanged. A worker guard
also forbids reading a private project that the user never identified, even if it
is the only available record or the assistant guessed it. User text deltas retain
their original whitespace. Two real worker-call regression checks reject invented
project identification and retain valid split-word user identification. Thirty
related worker/protocol/teacher/context/action tests pass; live control/model
behavior still requires fresh API recordings and independent audits.

**Plain words — eight updates completed:** 7435021 saves a checkpoint after all
eight joint updates (37.16GB peak GPU allocation), with frozen base hashes
unchanged. The subsequent fresh-process check stops before model loading because
its job-local driver directory already exists. Fix uses a separate reload driver
directory; a reload-only job can verify the existing checkpoint without training
it again. Saved SHA c4fc44a6fa17d363f7629bb202be1950f9a49ffab4ad1b0f5491a115558f9612.
No generated-speech or routing pass is inferred from these eight updates.

**Plain words — exact reload verified:** fresh-process job 7435228 reloads
7435021 with text/audio/policy max errors all 0.0. Native joint checkpoint SHA
c4fc44a6fa17d363f7629bb202be1950f9a49ffab4ad1b0f5491a115558f9612.
The eight-update implementation pilot is complete; behavior remains unqualified.
Real backend regression 7435106 completes all eight cases; its summary is to be
retrieved. Explicit control collection 7435049 records five scenes, then stops on
one worker-contract failure; dependent audit 7435108 is cancelled automatically.
This is a partial raw corpus, not approved supervision. Inspect received context
and improve error evidence before resuming only missing recordings.

A native-action replay is prepared: the learned head reads the previous temporal
state, with no scheduled trigger. A stable GPU action buffer is captured with
native graphs; the client reads it only between arrived audio frames. All action
probabilities and blocked repeats are retained. Actual workers use causal user
word context (diagnostic oracle, not live ASR), execute read-only tools and encode
only completed results. Cancelled/stale rows are rejected before application.
No assistant ground-truth words or oracle reference values feed generation.


## 2026-10-15 — Native checkpoint works technically, fails free-running behavior

**Plain words:** 128-update 7435379 completes on 46 TRAIN scenes with frozen
base hashes unchanged; fresh-process text/audio/action errors all 0.0. Checkpoint
SHA 44cb8b5b64838afd370f93fa2d2e9fc73bace43e5c971027941bf039ac9a547f;
37.49GB peak allocation. That does not qualify the model. Native replays
7435517/18/19/20/22 all fail on generated text/actions. Private-000 only says
"I'll check" without dispatch; Private-002 invents a circle; simple/clarify are
text-silent; cancellation repeats "record" without a successful cancellation.
Independent audio checks remain required. p95 native frame compute stays
30.45-35.60ms/80ms, but one private run has a 174.42ms buffered gap. No live pass.
Eight-update 7435380 also fails with no native dispatch and invented facts.

Backend-only regression 7435106 passes all 8 cases, 3 actual record reads.
Explicit client protocol yields 2/7 accepted control teachers in 7435484; other
5 remain rejected. Resumed failed recordings retain originals under incomplete/.
Conversion 7435530 and native encoding 7435538 verify both accepted controls,
max measured first-word error 100ms. New targets are 48 TRAIN total across three
hashed caches, with base validation/test folds unchanged. Next controlled tests
freeze language adaptation, compare baseline speech and isolate actual result
integration from the native head using an explicitly scheduled diagnostic.

**Plain words — controlled follow-up running:** frozen-language reference/action
training 7435667 uses all 48 approved TRAIN scenes; it must prove reference-free
speech unchanged exactly. Unadapted private/simple comparisons 7435670/7435669
and the trained scheduled-worker diagnostic 7435671 are running. The latter
explicitly does not test native routing. Independent audio audit payload records
silent clips as failed rather than aborting remaining clips. Lychee operator
profiling is prepared on its unchanged incremental runtime: two chunks after
warmup, no memo/cache/sampling changes. Profiled times are diagnostic, never a
latency acceptance benchmark. API follows the official PyTorch 2.9 profiler docs:
https://docs.pytorch.org/docs/2.9/profiler.html .

**Plain words — preserve speech and fix state mismatch:** frozen-language
7435667 completes 128 updates over 48 approved TRAIN scenes, training 540,675
new parameters. Reference-free text/audio errors before/after are both 0.0;
fresh-process text/audio/action reload errors are all 0.0. SHA
86dd746f0051f6b399b458b43462af25c7e6a5bc8b01892c864359a3da77da3e.
Native checks 7435824/25/26/27 and scheduled-result diagnostic 7435828 run next.
Unadapted simple 7435669 answers the question that 7435519 leaves silent,
confirming a language-adaptation regression. Scheduled trained 7435671 gives
wrong/incomplete facts despite an actual worker, so result use also fails.
Independent speech audit 7435717 confirms the spoken failures; near-silent clips
produce short Whisper hallucinations and WER 1.0, not evidence of actual answers.

Teacher-forced routing on 7435379 hits many validation action frames, while
free-running 7435517/19/20 never dispatch. This is a supplied-history mismatch,
not policy qualification. A native self-history collector is prepared: original
approved TRAIN user audio only, real sampled assistant text/audio, stable native
GPU hidden-state capture per 80ms frame, no worker references. Only existing
approved action clocks label states; generated words/audio are never speech
training labels. It must prove full frame coverage and retained source hashes
before a separate action-only fit can use it. No held-out audio is collected.

**Plain words — self-history and Lychee measurements:** native policy collection
7435849_0 passes all 501 frame/state checks on TRAIN-000, with exact model/voice
pins and sampled assistant history. No generated speech labels are approved.
The remaining 22 TRAIN scenes are submitted with at most four concurrent GPUs.
Lychee profiler 7435718 completes: two 400ms chunks show 40,733 cudaLaunchKernel
and 10,666 cuLaunchKernelEx calls, plus substantial repeated cuDNN attention setup
on single-query decode shapes. This is profiled diagnostic evidence, not latency
acceptance. A separate unprofiled experiment disables only cuDNN SDPA and compares
actual BF16 decode tensors against cuDNN and float32 math; no model, sampling or
cache change. Official backend/numerical reference:
https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html .

**Plain words — action-only fitting prepared:** native self-history states are
being collected for the 23 original TRAIN scenes (smoke 7435849 plus array
7435911). A separate 512-update linear action-head fit is prepared, using only
those captured states and approved control clocks. It leaves every speech and
reference parameter identical to frozen-language parent 7435667, checks parent
text/audio logits exactly, and requires another fresh-process reload. Neighboring
idle frames within 160ms of a single declared action are ignored, not relabeled;
no added positive actions, speech labels or word timestamps. Whole-task native
replays remain mandatory. Lychee attention-only comparison is job 7435951.

**Plain words — use the released low-latency backend:** Lychee without cuDNN SDPA
7435951 reduces token processing from 243.26s to 165.85s per 70s input, but still
has 97.49s backlog; not a realtime pass. Actual BF16 tensor probes differ from
cuDNN by at most .0078125 and have comparable float32 error. This does not prove
full-model trace parity or quality. The published Lychee realtime path is the
patched multi-stream vLLM backend, not the measured HF path; its README reports
about 2.96x speaking-round speedup. Thus HF slowness does not rule out Lychee.
Prepare an isolated ARM/GH200 build of the exact released patched vLLM source
against the working NGC Torch, without replacing Torch or altering accepted
model/source environments. The release expects Torch 2.5.1/vLLM 0.6.5; compatibility
with NGC 2.9/CUDA13 must be demonstrated, not assumed. A built wheel alone is
not enough: require native kernel execution and complete strict multi-head model
loading before any paced inference claim. Primary source is the pinned README
and third_party/vllm, commit 7bb1068c59b578e1f8e65fb762cd26708c14cc6d.

**Plain words — first vLLM attempt stops in source acquisition:** 7436166 cannot
materialize missing blobs from a local shared clone of the filtered asset repo;
setup.py is absent, so native compilation never starts. This is not an ARM/CUDA
compatibility verdict. Fetch the exact same commit from its actual upstream
promisor remote into an isolated sparse checkout and require build files before
entering the container. Existing accepted source/model environments are unchanged.

**Plain words — native self-history head improves actual action choice:** 7436033
fits only the action head on all 23 approved TRAIN self-histories. Speech/reference
weights match parent exactly; fixed-input text/audio errors 0.0 and fresh-process
text/audio/policy reload errors 0.0. SHA
11ce28b70c936be367e0cde24b85a2093e34412cb5e03feb3ae0f68fc1a85c8a.
Development 7436134 requests an actual yellow-circle read; 7436137 makes no worker
call on the simple question; 7436139's worker returns clarification without a
private read; 7436140 cancels native-0 with an actual receipt and rejects its late
result as stale. These are useful action-component results, not full-task passes.
Speech still gives wrong/private unsupported claims and premature cancellation
language. Backend context remains the causal measured-word oracle, not live ASR.

An explicitly disclosed spoken-text diagnostic is prepared. Only completed real
worker text may force native text slots while user audio continues through the
same native model. Manual inference word pacing is not a measured training label;
these outputs never enter training by default. This is hybrid text conditioning,
not evidence of learned ARC integration. Cancelled result tails are rejected by
actual receipt revisions. Three queue tests cover token order, partial cancellation
and atomic rejection of undefined words. Unmodified ARC mode remains default.

**Plain words — vLLM source now complete:** 7436322 reaches CMake, but nvcc cannot
write its inherited /local/user/... temporary path inside the container. Set a
job-local temporary directory under the already bound project filesystem. This
is still a build-environment failure before model/native-kernel qualification;
no evidence yet that the released runtime is incompatible with ARM/CUDA13.

**Plain words — actual worker words enter the native text stream:** hybrid
7436519 contains the real returned yellow-circle result, 7436520 contains the
actual clarification, and 7436521 contains cancellation confirmation after a real
receipt. None is a task pass: native speech invents private details beforehand,
misses later continuation, or makes unsupported claims. Independent speech audit
7436865 is submitted; generated token text alone does not prove intelligibility.

All previous native comparisons used the accepted generic teacher role. New
records-v1 instructions explicitly forbid invented private values/premature
completion, require clarification, and permit independent conversation while
waiting. The synthetic voice WAV is unchanged. This new role is restricted to
explicit worker-text diagnostics; ARC training prefix is not silently changed.
Native action heads must match the exact recorded role under which their
self-history states were collected. A scheduled private diagnostic can isolate
the role change while a new TRAIN-only policy-state collection is prepared.

**Plain words — CUDA13 native API correction:** 7436650 reaches native CUDA
compilation and fails on removed cub::Sum/Max operators. NVIDIA's CCCL3 migration
guide specifies cuda::std::plus, cuda::maximum and cuda::minimum replacements.
A recorded compatibility script checks the exact six original source hashes,
changes only these names/includes in the isolated build checkout, and records
after hashes plus diff SHA. It never modifies the accepted vendor checkout or
weights. Native numerical tests and strict complete model loading remain required.
Independent audio audit 7436865 rejects direct spoken-text forcing: WER .2941,
.0857 and .20; the returned private words are not reliably spoken as written.
These samples remain diagnostic and unapproved. The new role test 7437019 and
matching TRAIN-only policy-state collection 7437020 are running. No model passes.

**Plain words — test unspoken native lexical memory separately:** direct word
forcing disrupts the generated audio in the generic role. A separate diagnostic
adds each completed worker subword's frozen native text embedding to one streamed
temporal input. It never forces the model's output text/audio, never drops user
audio, and rejects cancelled tails. This uses the model's own vocabulary/basis,
not the unaligned ARC encoder, and has no learned integration weights. It remains
experimental and must pass actual spoken result use; no semantic compatibility
or qualification is assumed from adding embeddings. The accepted voice and
speech parameters stay fixed. New role prefix and native policy still need
matching re-evaluation; scheduled diagnostics remain explicitly marked.

**Plain words — role/policy and handoff:** records-v1 policy fit 7437285 saves and
fresh-process reloads exactly; parent fixed-input text/audio errors are 0.0.
Checkpoint SHA d54927fdf45567f974e3dfb65a882e69ab80962b1d061c5b7ec16ab431242f82.
Untrained lexical-memory 7437466/67 still fails to use the actual private value;
adding native embeddings alone is not enough. New-role direct-text 7437019 audibly
mentions yellow circle but inserts extra/changed words (WER .1622); still reject.

Next explicit hybrid test waits for five already-arrived quiet user and assistant
frames before starting completed worker text, then pauses that text on caller
speech. Every input sample still reaches the native model. Quiet uses measured
PCM energy, not future words; this handoff is an application rule, not learned
turn timing. Six native frames per inferred word are tested instead of four.
No such inference slots are approved word-time training labels. The goal is to
separate abrupt utterance interruption/pace from a fundamental speech decoder
limitation; unchanged immediate/four-frame controls remain available.

**Plain words — causal feedback experiment, 15 October:** quiet handoff trials
7437911/12/13/14 still fail content/receipt timing. The separate action head can
start a worker, but the speech transformer has no explicit pending/cancel state.
An optional zero-initialized six-state embedding now supplies only dispatched
client state and actually observed receipts. Chosen actions affect the following
frame, never their own neural input. Prefix state stays identically zero. The
bounded experiment warm-starts the generic-role self-history head 7436033, freezes
language adapters and policy weights, and trains the bridge/state inputs only.
It must preserve feedback-disabled speech exactly and fresh-process reload all
outputs exactly before native free-running tests. This is a hypothesis, not a
fix established by speech quality or task success.

A real label defect was also found: historical teacher cancellation 020 retained
generic delegate notes for both calls. Derivation v2 uses its strictly bounded
lookup-then-cancel chronology and actual receipt; a regression failed before the
fix and passes after. Immutable approved data is unchanged. Future conversions
retain the actual distinction directly. Existing checkpoints are not relabeled
as if they had trained with this correction.

Lychee build 7437283 compiles and links all 116 native targets after the recorded
CUDA13 reduction-name migration, then fails while copying FlashAttention Python
files into a missing package directory. This is a packaging failure; no native
kernel or complete model-loading qualification has passed yet.

**Plain words — feedback implementation pilot passed:** 7439916 completes eight
updates with finite nonzero state gradients, frozen base/language/action hashes,
zero prefix embedding, and feedback-disabled text/audio changes exactly 0.0.
Fresh-process reload text/audio/policy errors are all 0.0. Checkpoint SHA
`a0ddb339e713975e6eb5167ecbb40f4a7634f7a239c8d7be3a514bad7631d415`.
This proves implementation only. Native worker replay 7440182 and the bounded
128-update feedback stage 7440181 are launched. No behavioral pass is implied.

The next optional grounding objective contrasts already-approved different-value
TRAIN worlds for exactly the same original user question/project/field. It
predicts only the first measured private subword; the incorrect reference remains
negative conditioning, never positive speech. Both references retain their own
actual receipt clocks, and a pair is excluded if the negative stream is not fully
available before that decision. Nearby teacher assistant acoustic context is
masked with captured native prefix context in this branch, while user input is
unchanged. The model must prefer the real returned value over a different world;
teacher-forced margin improvement still requires native spoken holdouts.

**Plain words — state feedback alone does not qualify:** 128-update 7440181
reloads exactly (SHA 2383d56bb6303259acde6caacaa1a3de6bb53ce94228f06900971fed9c912e31).
All four native development replays 7440326/27/28/29 fail text/action evidence:
wrong private value, an unnecessary simple-question call, incomplete clarification,
and no successful native cancellation. Eight-update 7440182 also invents a
private value. No independent speech pass is inferred. Eight-update paired pilot
7440322 passes exact reload (SHA a641d25a99ee170c34d81da9722830e4f726d6e60ee1cc3ee7926e81e801a19e).
Its first private subword initially favors the wrong reference, confirming the
reference-use defect even on approved TRAIN material. This pilot is not accepted.

The next comparison preserves the self-history head and makes teacher-history
policy loss diagnostic (weight zero): its large teacher-forced loss otherwise
pushes new state inputs toward teacher feature states that differ from actual
self-history. Compare 128 updates with/without the paired grounding objective;
all other pins and approved sources remain the same. This is a new hypothesis,
not a claim that suppressing this loss resolves native routing or speech.

**Plain words — matched grounding stage:** 7440657 runs 128 updates with frozen
language/action weights and no teacher-history policy gradient; 7440658 adds
paired grounding with identical pins/data/seed and the same policy-loss setting.
Both require exact reload and new native development checks. Local causal-state,
paired-world, teacher approval, receipt, context and cancellation regressions pass
37/37. These unit checks do not grade model behavior.

The pinned Lychee config uses ten 40ms control slots per chunk, and its released
runtime buffers 400ms audio windows. Its default event-to-vocoder adapter flushes
25 speech tokens at a time. Token throughput alone will not establish conversational response latency. The
existing 200ms first-subword bar measures training alignment to actual word
onsets. Response-start timing, input backlog and streamed PCM must be measured
separately; no threshold is changed by porting the released backend.

**Plain words — released backend wheel now built:** 7439917 succeeds, wheel
`vllm-0.6.5+lychee.cu130-cp312-cp312-linux_aarch64.whl`, SHA
`1ed23f49f799377401882da464e0dfcc6fd9d53773a48d709a6ef8db5fa8bf0e`.
Kernel preflight 7440325 executes the native CUDA library but stops at FP32
1024-wide RMSNorm: maximum difference 4.77e-7 exceeds the initial two-epsilon
bound. The next check compares both native CUDA and PyTorch FP32 against an
FP64 reduction, using explicit FP32 absolute/relative 2e-6 tolerance; BF16/FP16
retain the two-epsilon magnitude-scaled bound. This checks ordinary parallel
reduction rounding, not speech quality; no model acceptance bar is changed.

**Plain words — native CUDA reduction tests passed:** 7440908 passes all nine
RMSNorm probes (512/1024/3584 widths, FP32/BF16/FP16) against both native-precision
PyTorch and FP64 reductions. Its subsequent patched-package import fails because
msgspec is absent. Full model loading, attention execution, speech and real-time
PCM remain untested. A separate CPU-only dependency overlay installs an explicit
subset of the pinned official lock with --no-deps; Torch/CUDA/Triton and the
accepted HF voice overlay remain untouched. Original failure evidence is kept.

**Plain words — both matched grounding candidates rejected:** native checks
7440909–7440913 (no contrast, 7440657) and 7440914–7440918 (paired, 7440658) fail
private facts/clarification/cancellation, and some ordinary answers degrade or go
silent. Their checkpoints reload exactly; implementation success is not behavior.
An always-on learned ready-state vector changes the whole dialogue even before
any dispatch. A new optional neutral-ready variant masks both prefix and ready
states to exact zero. Reference-free ready-state text/audio must remain exactly
unchanged; language and policy weights stay frozen. Simple examples become
retention diagnostics with no meaningless zero-gradient optimizer steps. Only
actual pending/result/cancellation states can learn feedback. This is a more
specific preservation invariant, not a claim it resolves result grounding.

Dependency setup 7441041 stops before installation because the bare container
has no TorchAudio distribution metadata (the voice environment supplies it in
an overlay). Protected-stack checks now preserve absent packages as absent;
no GPU dependency is installed or replaced to satisfy this bookkeeping check.

**Plain words — neutral-ready pilot preserves the intended baseline:** 7441177
passes eight updates, exact ready-state reference-free text/audio, frozen hashes,
and fresh-process text/audio/policy reload all 0.0. SHA
`4dd2e719871e4819f8010e3b99f4612063d0929765124a061949cfbb478db69a`.
128-update 7441316 and native checks 7441318/20/21/23/24 are submitted; no task
verdict is implied. CPU dependency 7441175 installs successfully without changing
the GPU stack. Backend preflight 7441242 then reveals the older HF overlay pulled
PyArrow26, which requires NumPy2, while the released vLLM lock requires NumPy1.26.
Add the same lock's PyArrow22 to the new isolated CPU subset; preserve the accepted
HF overlay and its working NumPy2/PyArrow26 environment. Failed dependent runtime
7441244 must not run as if its import proof passed.

**Plain words — Arrow pin resolved:** CPU overlay 7441408 succeeds. Preflight
7441410 again passes actual CUDA probes and progresses beyond the Arrow import,
then finds absent OpenAI SDK types imported by vLLM's generic chat utilities.
Add only the released lock's openai, jiter and distro CPU dependencies. This is
an import dependency, not a new provider call or a switch of the foreground model.
The blocked 7441411 dependent benchmark must be replaced after a passing import.

**Plain words — sampling comparison confound found:** the pinned exporter seeds
before loading the foreground. Native adapter/head construction initializes random
Linear weights and therefore advances CUDA sampling RNG even when their speech
effect is zero or immediately overwritten by a checkpoint. The self-history
collector uses the raw loader and does not consume those extra draws. New native
probes preserve CPU/CUDA RNG around construction/loading, record whether the
constructor advanced each RNG, and assert native RNG restoration. A disclosed
legacy mode retains the old behavior for comparison. Earlier outputs remain
failed evidence; they are not retrospectively labeled matched seeded controls.
This correction does not assert BF16 bit repeatability or solve task quality.

Neutral-ready 7441316 reloads exactly (SHA
694143ac54b15f3caa5519dafbacd35e871c9cfc6d3aab4347b01af9abcde86d).
Old-sampling native 7441318/20/21/23/24 still fail: private000 omits the real
value; private002 says green square after an actual orange-circle receipt; simple
content is wrong; clarification is absent; real cancellation occurs but spoken
confirmation/continuation remains inadequate. No model is promoted. The new
sampling-preserved comparison must precede conclusions about the accepted raw
voice/model versus the collected self-history policy.

**Plain words — patched backend Python import now passes:** CPU dependency
7441555 and preflight 7441556 succeed, with actual CUDA numerical probes and the
custom MultiHeadRequestState package import. Full-model job 7441557 reaches the
native integration loader and then stops on absent sentencepiece imported by
its generic GGUF loader. Add the released sentencepiece lock pin; no checkpoint
weights have yet been loaded or silently omitted. The successful import proof
is distinct from model/attention/speech acceptance.

**Plain words — timing terminology correction:** the earlier continuation text
incorrectly called 200ms a response-start target. `recovery_encode.native_word_timing`
checks each native first-subword frame against its independent measured word
onset (maximum offset 200ms). It does not measure the gap from user speech end
to assistant speech start. Historical live input-age/backlog gates are separate
again. Correct the new runtime metadata/docs so this training threshold is not
used as an invented end-to-end latency requirement. Actual response/PCM latency
still needs measurement against the desired conversational behavior.

**Plain words — released full-model backend runs at paced token rate:** 7441742
strictly accounts for all 982 checkpoint tensors: 488 native packed-model source
weights, 487 encoder, six audio adapter, and one explicitly unused merge token
embedding. All 329 native parameter/shard groups load, with no side-head fallback;
encoder/adapter state keys and shapes load strictly. Seventy seconds of arrived
input runs in 70.523 seconds, four responses/1304 speaking-token events; p50/p95
compute .278/.297 seconds per .400-second chunk. Initial preprocessing/encoder
startup produces max backlog 1.280 seconds. This is paced tokens, not live PCM,
voice quality, semantic success or microphone acceptance. New offline and
recorded-arrival streaming-vocoder checks preserve the accepted synthetic voice
and published lookahead/overlap rules; all returned codec values must be valid.

Future long jobs may select an immutable git worktree with DL_TASK_SOURCE and
DL_TASK_COMMIT; the payload verifies the exact clean commit and uses that path
for Python imports. This avoids queued jobs importing later main-checkout edits.
Submit from the main repo so existing Slurm log paths remain stable. Deploy code
through git only; artifact/log backups remain separate.

**Plain words — native speech decoding completed, streaming adapter import fixed:**
immutable-source offline vocoder 7443260 decodes the four native responses in
2.450 seconds (quality-only placement, not live PCM). Streaming replay 7443262/64
stops before decoding because the published adapter class is named
StreamingDecoder; the caller used an incorrect class name. Correct that exact
import and retain both failures. The retry must also account for every native
codec token exactly once, including any actually interrupted unplayed tail.

Sampling-preserved PersonaPlex 7441744–53 confirms constructor CUDA RNG advance
and successful restoration. Task speech still fails private values, clarification
and cancellation; do not promote the neutral-ready checkpoint. The paired TRAIN
objective improves 16/16 reference margins (mean 4.723) while native task behavior
still fails, another direct teacher-history/free-run distinction. Prioritize the
now-operational Qwen-based Lychee path: test actual completed worker text in its
nonspoken instruction context, with every already-arrived user-audio cache and
native speech/control token preserved. First use scheduled dispatch only as an
explicit result-integration diagnostic, then require learned native dispatch.
No word forcing, oracle private values or held-out training is authorized by
that diagnostic. This interface is proposed and still needs a preservation proof.

**Plain words — actual-text context interface prepared:** an inference-only
Lychee diagnostic keeps the approved user waveform and every native text,
acoustic and control token, while rebuilding only its nonspoken instruction
prefix after an actual pending dispatch or completed receipt. Existing audio
features/tail, next audio clock, relative speech mapping and RNG must survive
exactly; unsafe text-only KV prefix caching is rejected. A paired different-value
world test will check actual response dependence. Dispatch is scheduled and
explicitly not a learned foreground choice. Backend context remains the measured
word oracle. No assistant word/audio output is forced; no validation data enters
training. The next runs must establish preservation and spoken correctness.
The published constrained-model sampling path is recorded and its actual audio/
control outputs are required at every step; missing outputs cause an error
instead of an undocumented alternate-token fallback.

**Plain words — first native context interface preserves state but not result use:**
7443841/42 use real paired-world reads and pass all history/audio/RNG preservation
checks with zero alternate side-token fallbacks. The model keeps saying the
lookup is pending instead of reliably stating its completed value. Context
preservation alone is insufficient. In these early report files the field
actual_worker_results_used means supplied into context, not verified spoken use;
new reports say supplied and explicitly leave spoken use unverified.

Test clearer current-state language in place of JSON/timestamps in the nonspoken
prefix: distinguish a currently pending lookup from a now-completed authoritative
record, and explicitly mark earlier lack-of-access statements outdated. Both
worlds must still depend on actual returned values. Output words, acoustic tokens
and speaking-control state remain sampled; no response-start override is added.

**Plain words — explicit completion still not spoken:** 7448884/87 again pass
history/audio/RNG preservation and actual audio/control output audits, but omit
both completed private values. The base invents an authorisation requirement in
its refusal. The next matched diagnostic clarifies the already-authorised
read-only synthetic exercise and compares static supplied reference against
actual late worker receipt, with the same role wording. Static reference is
explicitly oracle-at-start, performs no worker call, and can only test whether
the base reads context; it cannot qualify delegation or causal result integration.
Actual mode still uses real receipts and sampled output/control tokens. This
separates reference comprehension from late-result response/refresh behavior.

**Plain words — static reference works, actual late result still omitted:**
7450548/50 correctly generate yellow circle and purple diamond respectively
when supplied at the start (explicit oracle controls). Actual late-return
7450551/53 still omit both values. The model can read/speak supplied context;
causal continuation needs further diagnosis. A disclosed client-response event
will force one allowed future start-speaking control after a real result when
already-arrived user audio is quiet. No past token, word or acoustic output is
forced. This is application-controlled timing, not learned model turn-taking or
foreground delegation qualification. Both actual worlds must be checked.

**Plain words — reproducible playback defect fixed:** a test of two separated
utterances reproduces the old one-time-buffer defect (160 spurious gap samples
in the second utterance). Buffering every utterance fixes it; seven playout
regressions pass, including genuine stall visibility, cancellation of queued old
PCM, and preservation of unrelated speech. Actual hop8 trace 7443647 reproduces
its original WAV exactly before re-rendering. Fixed per-utterance 240ms buffering
and recorded native interruption stops render 1,155,920 samples, cancel 26,800
queued samples before playback, and leave zero within-utterance underrun samples.
Every source sample is accounted for. Artifact: runs/lychee-playout-7443647-buffer240.
This is a recorded component replay; concurrent LM/vocoder/transport and human
voice quality remain unverified. Full-transcript WER can include intentionally
cancelled tails, so distinguish that from actual text/audio disagreement.

**Plain words — startup input warmup prepared:** the first native paced trace
spends 1.141s in first-call mel preprocessing and 0.302s in the first audio encoder
call. An optional pre-user warmup processes only 400ms of zeros through those
front-end functions. It never creates a conversation, reads user samples or
generates LM tokens; CPU/CUDA RNG and absence of an active request are checked.
The warmed baseline must measure actual backlog again rather than subtracting
startup time from an existing result. No cached user/audio prefix is reused.


**15 October, result-integration diagnosis:** static reference controls
7450548/50 correctly say both private values. Actual late receipts 7450551/53
omit them, and the disclosed application response-start controls 7450827/28
also omit them. This rules out a missing response-start event as a sufficient
fix. New diagnostics audit the exact fresh system prefix at the actual vLLM
add-request boundary and compare immediate delivery with delivery when the
assistant is listening and two already-arrived 400ms user chunks are quiet.
The latter is explicitly application scheduling; all words/acoustics remain
sampled. Keep backend-ready, context-delivery and response clocks separate.
Neither variation enters training. Warm-input benchmark 7451709 is submitted.

Per-utterance playout replay is now verified: fixed 240ms buffering produces
zero within-utterance gaps for trace 7443647, preserves every played PCM sample,
and cancels 26,800 queued samples at actual native interruption times. This is
component replay only. ASR audit 7450830 gives full-transcript WER 0.2763 for
both offline and repaired playout; intentionally cancelled tails and spoken
number formatting need separate analysis. This does not pass the 0.08 bar.

Two local regressions pass: a stale system prefix is rejected before engine
submission, and quiet user audio alone cannot trigger idle delivery while the
assistant is speaking. These exercise instrumentation/scheduling, not model
result use. No inference result for the new variants is claimed yet.


**15 October, actual engine boundary checked:** 7451814/15 (immediate) and
7451816/17 (idle/quiet delivery) verify both pending and completed system prefixes
at the real vLLM add-request boundary. All four still omit the actual returned
value. Both quiet-delivery runs hold the completed result until 29.2s, adding
19.8–20.1s. Do not adopt that delay or repeat prompt/timing-only variants as a fix.
Native result-integration adaptation and a native HF/vLLM forward comparison are
next; static reading success does not establish late-result competence.

Warmed baseline 7451709 completes 70s of paced input in 70.521s, p95 compute
301ms/400ms, maximum input backlog 18.5ms (previously 1.280s). All text, acoustic
and control token arrays are exactly equal to original 7441742. Warmup itself
takes 1.339s before input, preserving sampling RNG and creating no conversation.

Concurrent LM/PCM capture is now implemented for two separately assigned GPUs,
with isolated dependency overlays, a bounded local socket/queue, and independent
receipt of actual interruption controls while the vocoder is busy. It measures
PCM availability during actual LM execution. The final WAV simulates fixed
240ms causal playout; no audio device/browser/microphone acceptance is implied.
Every native codec unit must be consumed or explicitly cancelled. Startup,
transport, bounded-queue and child-process failures are fatal. Eight local socket,
delivery and playout regressions pass. The actual concurrent GPU job is next.

**15 October — concurrent capture and graph comparison:** concurrent two-GPU
job 7452017 is submitted from immutable c2bb062. No outcome claimed yet.
A separate optional native prefill capture preserves actual full input embeddings,
raw three-head logits and the actually sampled next text for merge conditioning.
It does not edit output tokens or RNG. Synchronous projection/copies invalidate
its timing figures. Three captured fresh prefills (including speech history)
will be replayed through the strictly loaded released HF training graph, with
identical embeddings and historical/current next-text merge conditioning.
Report full-vocabulary max/RMS error, KL, top-one agreement and top-20 overlap;
do not fit an acceptance tolerance after seeing errors. Shared input embeddings
isolate graph/branch/head behavior; this does not validate the input encoder or
incremental KV. No optimizer run or new training checkpoint exists yet.

**Plain words — native graph and concurrent audio results:** capture 7452224 /
HF comparison 7452225 match all nine raw-head top-one predictions at 11/440/840
positions, with strict full HF checkpoint loading. Maximum RMS logit error
0.1219, KL 0.00863; no fitted tolerance or numerical identity claim. Shared input
embeddings isolate the graph, not the audio frontend/incremental KV. Actual
control masking is not part of the full-vocabulary top-one comparison.
Concurrent 7452017 verifies exact baseline native token equality, real concurrent
PCM availability, zero within-utterance gaps with fixed 240ms playout and exact
1258-unit accounting. Its old nested vocoder_mode string is a stale hardcoded
label; parent/receiver evidence proves concurrency, and later code corrects that
label without changing the immutable artifact. Owner listening is requested on
the refreshed page while independent work continues.

**Plain words — native S3 data stage prepared:** only the 48 approved TRAIN
scenarios can be re-encoded. Actual measured gaps split speech segments; >28s
continuous segments are explicitly excluded in this bounded codec diagnostic.
Every source word interval/hash and actual result receipt is retained, and the
native tokenizer/vocoder weights and accepted reference are pinned. All outputs
remain training_approved=false until independent S3 speech auditing AND causal
native assembly checks. Source/fold/word validity failures are fatal. Two source
word-bound regressions pass. No new data are invented; no validation/test audio
is processed. Native semantic adaptation should begin with layers 24–27 only,
after the speech/control branch points, subject to an actual preservation proof.


**Plain words — native data stage completed:** S3 conversion 7452500 reads
48 approved TRAIN scenarios and creates 125 measured speech segments. Independent
round-trip audit 7452502 accepts 106 under WER <=0.08 and rejects 19; no threshold
is relaxed. These are codec-intelligibility results, not approved native training
windows or task success. Keep failed segments excluded. Native model histories
will be collected using actual workers and sampled control from approved private
TRAIN scenes, preserving both the ready moment and the first subsequent quiet
native listening boundary. Snapshots retain all actual sampled body tokens,
arrived audio features/embeddings/tail and actual receipt clocks. They contain
no gold future labels yet. Validation/test history is rejected before tensor
access; future/unreceived results are rejected. State collection invalidates
latency qualification. Two collection-boundary regressions pass.


**Plain words — actual student history ready for native semantic assembly:**
7452624–31 complete all eight approved private TRAIN scenes. Every run stores an
actual completed-result snapshot and a later native listening snapshot, with
all sampled student history and received audio preserved. All omit the value;
these are useful error contexts, not successful candidates. Native replay now
captures the actual complete prefill on the next arrived user chunk, then
appends a verified gold reply only where all remaining real user audio is quiet.
The quiet extension is an explicit training augmentation, never an inference
result. Native S-S/detect/sleep/K-S/S-L grammar and all S3 units must round-trip
through the native decoder exactly. Only future words and EOT receive loss;
all old history and speech/control labels are masked. Every first target is
after the actual receipt and the snapshot clock. Gold speech must have exact
independent transcript agreement (stronger than the 0.08 corpus bar) and contain
the actual returned value. Missing or failed gold segments remain excluded.
A layout regression checks future timing, native boundaries and codec retention.
Native GPU assembly and an adapter gradient/preservation/reload proof are next.


**Plain words — native semantic inputs passed:** 7452831 builds seven causal
TRAIN examples from the actual collected native listening histories. The eighth
source has no exactly transcribed native gold reply and stays excluded. Actual
prefills contain 759–979 tokens; 8–9 future text/EOT targets per example begin at
25.2–34.8s, after the corresponding real worker returns at 8.8–11.6s. Native
word/code/control decoding reproduces each gold reply and all codec units exactly.
The quiet future is explicitly an augmentation, not an inference result. No
optimizer ran in this assembly.

A bounded eight-update semantic LoRA proof is prepared on main layers 24–27
only, after speech/control fork at 24/20. It freezes every base parameter and
uses only future semantic text loss on approved native inputs. It must establish
zero-adapter identity, preserved CPU/CUDA construction RNG, finite nonzero
gradients in all intended tensors, identical frozen parameter hashes, unchanged
raw speech/control hidden outputs for the same input, changed text logits and
exact fresh-adapter reload. Every data consumer revalidates source quality and
hashes of native units, independent audits and actual student histories. A pass
would prove implementation only, not task performance or merged native deployment.

**Plain words — native semantic proof stopped safely:** 7452899 loads the
complete HF model and passes zero-adapter equality, then rejects a missing or
nonfinite gradient before the first optimizer step. No checkpoint is produced.
The next diagnostic identifies the exact tensor and missing/NaN/Inf counts;
no relaxed gradient or quality condition is introduced. Explicit native weight
merging is also prepared but cannot consume an unproved checkpoint. It verifies
all 28 post-branch matrices, records BF16 rounding and does not claim exact
parity with dynamic adapters. No new native task result exists yet.

**Plain words — gradient failure reproduced:** 7452921 identifies NaNs in all
28,672 entries of layer24 q_proj A at finite CE 1.95237; the gradient exists.
No optimizer step occurs. Matched diagnostics will change only SDPA kernel
selection: cuDNN disabled, or FP32-accumulating math attention. Both retain all
zero/frozen/gradient/reload checks and the same data/seed/rank/lr. Default failure
is preserved. The backend must be recorded in any checkpoint; no nonfinite
replacement or relaxed proof gate is allowed.
