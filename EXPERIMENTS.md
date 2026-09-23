# Phase-1 ablation ledger

One row per training run: the recipe, the held-out exam result, the
verdict. **Keep this current with every run** — it is the memory of what
we tried so we never re-buy the same lesson. (Plain-language rule
applies: acc = how often the model's top pick was the right token; prob =
how much of its confidence it put on the right token, on average.)

**What "the exam" is:** every run trains on 80 factory lessons, then is
quizzed on 8 lessons it NEVER saw (held-out). A run only counts if it
passes on unseen lessons — that is generalization, not memorization.

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
