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

## 17 October 2026 — matched controls and a new corrective stage

Plain words: more updates alone did not solve conversation. Keep balanced136
checkpoint7500125 as an unqualified reference; reject unchanged6487500366 for
wrong explanations/repetition. Cancellation5207500371 learns a useful backend
request on validation007 but falsely denies authority in speech. The new stage
adds the first clarification question and cancellation controls on a newer actual
TRAIN history, starting from136-update weights with an explicitly fresh optimizer.
These are development checks, not the historical80-lesson Phase-1 exam.

| Training run | Recipe and engineering result | Native verdict |
| --- | --- | --- |
|7500012/13,7500117/25| Matched14,8 then128 updates; component-order vs equal behavior-kind sampling; exact optimizer/random-state/prediction reload | Balanced private values spoken in three cases; full tasks fail. |
|7500366|512 more updates from7500125,648 total; exact reload, lower text loss | Reject wrong facts, repetition and missing values7500380–91. |
|7500364/7500371| New15 components including isolated cancellation controls;8 plus512 updates, exact reload | Validation7500397 requests after cancellation and backend executes it, but speech denies authority; TRAIN7500396 misses the request. Unqualified. |
|7501360| New17-component warm8 from7500125 weights, optimizer reset; independent audit7501359 and exact state/prediction reload PASS | Completed private/ordinary/clarification/cancellation checks fail full tasks. No unchanged continuation. |
|7501930/7502015| New18-component recipe adds50 measured post-answer listening controls; warm8 plus128 updates pass exact reload | Completed native7502018–29 fail full tasks: wrong explanations, unnecessary ordinary requests, no clarification or cancellation; some private values in sampled text. Reject unchanged expansion. |
|7502129/30 (inference only)|16 silent warm-up frames followed by full reset; identical generated packet audio and text | Maximum input backlog3.1746s cold versus0.3938s warm; one paired runtime gain, no answer-quality gain or training labels. |

Plain words: detail, diagnostic IDs, rejected rendering and corpus ownership live
in [the current control-recovery record](docs/research/2026-10-17-control-recovery.md).
The ordinary three TRAIN checks fail, the released persona does not repair the
completed comparisons, and BayLing's bounded unchanged audition does not justify
a backbone switch. No complete native model or cloud GPU resume is qualified.

## 16 October 2026 — actual-history corrections, still no qualified model


Plain words: corrective ordinary-answer preparation is now sourcec35f8fb.
Observe-only mode records natural simple TRAIN histories without backend
execution or answer-bearing spans; it cannot qualify a task or approve labels.
Captures7486102/04/05/06 correspond to TRAIN008/009/010/011, with exact zero-proof
voice/layout7463825 and expected-kind=simple. First validate all four with
`evals.context_history_validate` at collector commit
`c35f8fbbcfa402ad43683d474ab66a1ac39ce292`, then add separate ordinary-answer
future eligibility. Private receipt checks must remain unchanged. Metadata
feasibility of the old four raw histories found answer-bearing spans during
three teacher replies; report is `simple-control-feasibility.json` in the local
continuation directory. New captures are raw conditioning only, not training.


Plain words: early17 October — targeted state/payload tests finished. Native
pending-status runs7485621/22 produce the correct orange square/teal star;
independent CPU speech audit7485697 confirms both complete values (word error
against sampled text0.0606/0.0270). No spoken text/audio/start is forced. Without
pending status7485619/20 still invent D138/six. Receipt-only changes do not
establish safe waiting. This is improved private-result behavior, not complete
conversation qualification: TRAIN000 still omits the intervening physics answer.

Plain words: broader pending-state controls remain failures.7485698 gives correct
pencil/graphite text;7485699 is incoherent about evaporation and7485700 ignores
cancellation. Actual cancel-scene backend requests see only audio BEFORE the
cancellation, so no cancellation tool is dispatched. Zero-adapter baselines
7485702/7485755 also fail the private cases; ASR/backend timing differs, so this
does not isolate the contribution of training.7485701 was rejected for missing
TRAIN-scene authorization flag and is not a model result;7485755 is its retry.

Plain words: do not expand the failed private-only recipe. Its13 new supervised
continuations are private facts; other behavior kinds are base-distribution
retention, not corrective supervision. Next training work must explicitly
approve ordinary/clarification/cancellation request/timing targets from TRAIN
sources and actual arrived user speech, with separate eligibility rules for
non-factual actions. The fact-only receipt gate must not be weakened to smuggle
in unsupported control labels. In parallel, measure request delay, pre-receipt
speech and post-receipt value use separately. No qualified native model exists.


Plain words: late16 October — owner explicitly prioritises Isambard model gains,
not extra compute. Cloud validation is ON HOLD; $25 remains approved for later
validation only, with zero cloud resources launched. Clifton was renewed and
SSH works. Both136-update13-case continuations7480819/20 complete and reload
exactly but do not pass native controls; do not expand on text loss alone.

Plain words: a concrete backend interface bug hid actual successful receipts
when worker prose answered only a later question (7480841: lookup=`teal star`,
injected prose=physics only). Source3e42038 now injects verified actual record
facts alongside prose; stale/fabricated/cancelled receipts cannot be promoted,
and errors remain failure notices. All21 focused regression tests pass.
The incorrect numeral in7480830 completes at9.957s before real receipt injection
at10.869s: waiting and available-result use are distinct failure mechanisms.

Plain words: four targeted native tests7485619–22 keep the136-update own-audio
weights fixed. Receipt preservation alone still fails. Adding actual pending
status (no forced speech) yields correct sampled orange square/teal star text
in7485621/22 without the earlier fabricated values. This is preliminary text
behavior, not full voice/task qualification. Independent CPU speech audit7485697,
ordinary/clarify/cancel controls7485698/99/5700 and zero-adapter comparisons
7485701/02 are running/queued. Exact payload events and phrase/receipt clocks
are saved under `runs/recovery-continuation-20261016`.
See `docs/research/2026-10-15-resource-decisions.md` for what was actually adopted
from each supplied resource and what remains unimplemented.


Plain words: latest checkpoint at19:45Berlin supersedes transfer/pending claims
below. Both full public model weights are exported locally and read-back SHA
verified (33.49GB total), with `runs/cloud-export-20261016/model-verification.json`.
All25 corrected synthetic histories are backed up in the workspace/export tree;
four critical file hashes per case pass. They remain conditioning-only, not
optimizer-approved. Required zero/layout proof7463825 is included too.

Plain words: the provider-neutral amd64 image builds and runs CPU state/data
checks. Actual Isambard136-update checkpoint7475992 deserializes on amd64:
26 finite adapter tensors, finite optimizer moments, CPU/Python/NumPy random
states accepted. CUDA-state restore/forward/backward on a destination GPU is
STILL UNTESTED. No cloud resources provisioned; provider/account and spending
cap remain undecided. Use `portable/context/README.md` to restore and validate.

Plain words: matched native tests7479192–97 all fail task qualification, including
private TRAIN/validation, simple, clarification and cancellation. They use the
correct fixed voice/layout, so this is a legitimate rejection for scaling the
three-conversation136-update recipe. New loader789d3f3 restores the checkpoint's
required native layout and MATH attention automatically, exactly once;7480107
runs without explicit layout/math flags and still gives the wrong private value.
All51 saved prefix tensors match the explicit-protocol run7479192 exactly;
this verifies that prefix only, not the complete generated trajectory.

Plain words: corrected25-case collection7479198 completed. Eligibility report
`context-matched25-eligibility-9c5dc2047dc4019a6f39a5efc10894fdfe2586fa`
approves13 cases for gated fact-future capture, not for optimization. It rejects
unsupported ordinary/cancellation targets and cases without a valid silent
prefix/measured sentence/actual matching receipt. All13 teacher-audio future
captures7479955 completed. Independent temporal audit7480420 PASSES all13;
maximum streamed/batched error0.0008345 (limit0.001), causal prefix error0.
Corrected eight-update/reload proof7480562 PASSES, frozen base/speech unchanged.
Real job IDs:7480269,7480270,7480273,7480275,7480280,7480281,7480282,
7480283,7480289,7480290,7480292,7480295,7479955. Do not use the old7477252 map.

Plain words: native-audio futures7479546/47 pass independent ownership/delay
and TRAIN-only audits. Assistant audio is naturally sampled; approved teacher
TEXT is forced, and real source receipts are replayed. CPU speech audit7480258
recovers the intended Purple Triangle/orange triangle values, with word errors.
This is a rendering diagnostic, not native reasoning or live delegation success.
All13 native-audio counterparts7480417 completed with restart gates passing,
sourcee150524. Independent temporal audit7480482 PASSES all13, maximum error
0.0002728 and causal prefix error0. Corrected eight-update/reload proof7480563
PASSES exact predictions/optimizer/RNG reload, frozen base/speech unchanged.
CPU speech check7480484 recovers11/13 complete private phrases; mean word error
10.16%, median8.11%. Two Blue Circle renditions are incomplete/misrendered despite
supplied text. This is not native policy success or a human voice-quality gate.
Native real job IDs:7480443,7480444,7480445,7480446,7480447,7480449,7480454,
7480455,7480463,7480464,7480465,7480466,7480417. Interactive first-source7479666
remains pending and is not evidence of faster scheduling. Both new checkpoint
states/data/adapter hashes are verified in the local export. Proof launch7480433
was rejected before optimization because CONTEXT_ACTUAL_TEMPORAL_FIT=1 was
omitted;7480483 was cancelled pending to fix the same launch flag. Corrected
jobs7480562/63 are the valid proofs. No training source/data rule was weakened.

Plain words: paired128-update continuations7480819 (teacher acoustics) and
7480820 (own acoustics) resume their exact new eight-update parents, sourcee150524.
Native tests depend on successful completion/reload: teacher7480830/32/33/34/35/36;
own7480837/39/41/42/43/44. Both test TRAIN000, changed private validation000/001
(silver hexagon/teal star), pencil005, clarification006 and cancellation007.
These are known audio scenarios with new private worlds, not unseen audio.
There is still NO qualified native checkpoint. The paired baseline text losses
are0.716 teacher-audio versus0.921 own-audio (FP32, same13cases); this is a
conditioning diagnostic, not a proven explanation of native task failures. New data/source is a new stage, never a silent exact resume.


Plain words: corrected loss run7475992 reaches136 updates and reloads exactly,
but the native task failures remain. TRAIN000 and changed private validation
000/001 do not give complete correct values; pencil005 text is correct;
clarify006 fails; cancel007 hits an invalid-tool argument. Reject scaling this
three-conversation recipe. Source90ef8eb preserves invalid calls as failed
backend results instead of crashing PCM, with16 passing regression tests.
The next raw-data collection targets actual TRAIN simple008/009, clarify016,
cancel020 (7476737–40). Earlier7476271/76/79/81 were private004–007, not controls;
3005c58 adds an expected-kind guard. No new raw history is an approved label.

Plain words: a source comparison identifies an actual training mismatch:
upstream includes PAD3 silence with0.5 weight, while our early continuation
mask discarded it. New sourcead82745 retains capture verification unchanged,
approves all real future text frames and weights silence as upstream does.
Audit7475863 passes257/257/240 targets, including169/181/172 PAD3 targets; no
past/span/user/audio loss. New base8-update7475934 passes exact separate model,
optimizer and RNG reload. Word and silence metrics are reported separately.
Do not compare its blended loss directly with the old sparse-word objective.
Old40-update development private cases all fail complete changed-value use;
old168-update7475556 improves teacher-conditioned prediction but proves no task
success. No repeated silent/gold/forced-result behavior is accepted as native.

Plain words: corrected early audits approve88/76/68 text targets. The complete
continuation needs measured per-word positions to locate the first fact;
whole-value tokenization can differ at punctuation. Corrected source0e0e363
passes numerical audit7469530 on all3 components: largest text error0.00008321,
causal prefix error0. Eight-update7474819 passes actual model, optimizer and RNG
reload. Native7475190 still gives a wrong/partial private answer. Resumed run
7475099 adds32 updates (40 total), passes separate reload and keeps base weights
unchanged; training word prediction improves but free-running success remains
unproven. Development batch7475396/98/5400/02/04/06/08/10 is submitted.
Custom checkpoints are backed locally with hashes checked. No native model is
qualified by these engineering or teacher-conditioned scores.

Plain words: the complete early continuation now starts before the student's
first reply, retaining the original user audio and result-arrival clock. Three
native captures7467981/7468102/7468611 completed; first independent audit passes
88 text targets with zero past/span/audio losses. Temporal audit7469288 is
pending; no training or free-running improvement is inferred from captures.
The new checkpoint helper also passes exact interrupted-versus-uninterrupted
CPU continuation and rejects changed data. Full-model optimizer-state resume
and amd64 GPU continuation remain unverified. Provider-neutral image recipe is
in `portable/context`; no paid cloud resources have been provisioned.

Plain words: training on complete teacher conversations improved prediction on
those conversations while damaging free-running behavior. We now keep the actual
student conversation as conditioning and add only a separately approved teacher
sentence after a verified quiet boundary and delivered matching private result.
Exact full native Engine restart is required before creating any future targets.
This changes the training data path; it does not establish a useful model.

- Components7466157/60/63: actual TRAIN histories7465804/06/07, exact restart
  through327/224/259 real frames. Future lengths32/56/35; eligible text labels
  12/23/21. Past and audio loss labels both zero. History001 already said the
  correct value and is excluded. Full hashes, builder/source commits, approved
  teacher lineage, user PCM, measured word spacing and real receipt checked.
- Head-only proof7466440:8 updates; exact fresh-process FP32 and native-BF16
  feature logits; frozen parameters unchanged. Adapter SHA
  `0c70dde5983d272a7c3772f23808a7c50fff6f2487642d0b573a807584ca6563`.
  Native text negative log probability (lower is better)1.819252→1.819245;
  first fact-token accuracy was already3/3 under earlier teacher speech.
  Verdict: valid engineering proof, negligible fit gain, no result-use claim.
- Zero-update temporal audit7467209: all3 pass streamed/batched FP32 MATH
  comparison, maximum text errors0.00004745/0.00004911/0.00005651 versus
  threshold0.001; all causal prefix errors0. Captured BF16 versus FP32 hidden
  differences0.346/0.284/0.238 are separate diagnostics, not relaxed native
  restart gates. Already-delayed inputs are never delayed a second time.
- Actual temporal proof7467496:8 native-BF16 updates, last4 rank8 temporal
  adapters plus text head; independent approved TRAIN retention examples from
  all4 kinds. Retention penalizes changes to the base text distribution (KL,
  a distribution-distance measure) and internal states (10x mean squared
  difference). All actual past/audio labels absent, base speech weights frozen.
  Updates finish; peak GPU17,633,427,968 bytes. Check full final reload and
  native7467526 before accepting any behavioral improvement. No held-out exam
  or voice-quality pass follows from fitting3 TRAIN corrections.

All3 components, source histories, head proof and temporal audit are backed
locally with hashes checked. Full logs and submission receipts are in
`runs/recovery-continuation-20261016`. Native head-only development batch
7467049/51/53/55/57/59/62/64 remains in workq. Short engineering checks use the
documented interactive reservation; routine batch work stays in workq.

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

**Plain words — collect counterfactual native histories independently:** the
13 previously approved paired-private TRAIN teachers can use the same guarded
actual-worker collection through their own conversion receipt. This exposes the
native student to actual different returned values for shared questions, rather
than treating reference-reading or seven repeated facts as general grounding.
No source/counterfactual/held-out data are relabelled. Native S3/gold and causal
assembly checks still decide which collected histories may train. This read-only
collection can proceed while the gradient-kernel diagnostics run.

**Plain words — native semantic implementation now passes:** no-cuDNN 7452937
and math 7452938 both complete eight updates on identical data/seed/recipe. Both
train all 56 intended tensors (2,883,584 parameters), preserve zero-adapter text /
speech / control exactly, preserve frozen SHA 97006db0d1e74178277b1c474f5bb85200c40d7c9e36997a073369d475e77098,
preserve same-input raw speech/control outputs exactly after updates, and reload
fresh adapters exactly. Default 7452899/7452921 fail before updates; same-GPU
kernel diagnosis can further separate backend from node differences. Use the
no-cuDNN checkpoint as the first native deployment diagnostic, not a qualified
model: SHA 5c238a774042eee2596350b53fd04afd83ea689b8ed470475001e7636adf7965.
The math checkpoint SHA is 6e01535ec100429936ce1cac4ee2c22477e09a115d0926ecbcbf8697b7498d5d.
Cold native BF16 merging must account for all 28 intended weight matrices and
retain the strict all-982-source-tensor loading receipt. Run both actual worlds
with native/model timing and with the disclosed single client-response event.
All words/audio remain sampled; dispatch is scheduled. No task success claimed.

**Plain words — attention diagnosis within one GPU prepared:** optional first-
backward comparison runs default, no-cuDNN and math on the same zero-initialized
adapters, same data and same allocated GPU, without any optimizer step. CPU
operator profiling records the actual selected SDPA operations and backward
names. Every bad gradient remains recorded; all gradients are then cleared and
the chosen finite backend must pass the original eight-update proof. This
separates kernel availability from the cross-node difference in earlier runs.


**Plain words — eight updates do not fix late results:** cold native
7452981/83 (model timing) and 7452984/85 (one disclosed response event) still
omit yellow circle and purple diamond. All native merges and strict loading
complete; wording changes but no task improvement is claimed. This was one
implementation-proof pass over seven examples, not a trained release. Do not
promote it. Paired native history collection 7452946–58 completes all 13 actual
counterfactual TRAIN worlds. The next bounded behavior pilot will use those
additional histories only where native codec/gold and causal assembly checks
pass, before comparing a finite number of training lengths. Actual ordinary
answers and speech must also be rechecked before promotion.


**Plain words — same-GPU backward cause isolated:** 7453014 uses nid010035,
GH200 120GB, Torch 2.9.0a0+145a3a7bda.nv25.10, CUDA13 and cuDNN91400. The
same zero adapters/input on that GPU give 44 invalid-gradient tensors through
aten::_scaled_dot_product_cudnn_attention_backward; efficient attention and
math give zero. CPU profiling confirms actual selected operators. Clearing
probe gradients and training with cuDNN SDPA disabled reproduces all eight
losses and checkpoint SHA exactly from 7452937. Training now defaults to the
verified efficient path. Serving still uses its separate native custom backend.

Paired assembly 7453119 passes 15 causal native examples from 21 collected
TRAIN histories; six without exact native gold transcripts remain excluded.
A bounded pilot permits only 64/128 updates after the passing implementation
proof, rechecking every original source approval and exact frozen base hash.
Both arms retain zero/frozen/branch/gradient/reload gates and cannot qualify
quality from loss. Their native actual-worker replays and ordinary answers
must decide whether this recipe improves the task. No corpus-scale run is
justified yet; no failed-gradient checkpoint is allowed.

**Plain words — paired semantic pilots completed, behavior still untested:**
7453182 (64 updates) / 7453183 (128) pass every native implementation gate
on 15 verified examples. Their first 64 losses match exactly. Both keep the
same frozen SHA, zero same-input raw speech/control drift, all finite/nonzero
adapter gradients and exact fresh-adapter reload. Checkpoints:
64 = 41c627145683a514981ce04e0471af5228cd4ad0ccd22cfd27d49b2a79bb001f;
128 = f1be05e12c92ee74e54c0225d79dd3c602ed7a179a2bb626e448ca0cd2e1a892.
No quality or native result-use success follows from low training loss. Next
are matched actual-worker worlds under native and disclosed response-event
timing, plus unchanged ordinary science inputs for each arm.

Concurrent transport now supports that same actual-worker path with a shared
monotonic input epoch, arrived-audio context refresh and optional explicit
semantic checkpoint. The parent records real receipts and distinguishes its
scheduled dispatch/forced response control from native model choice. Only the
LM child receives the API credential. No concurrent-worker outcome is claimed
until the two-GPU job completes; this is not a microphone/browser trial.


**Plain words — both longer semantic pilots fail actual tasks:** native
7453230/38/39/40 (64) and 7453242/43/44/45 (128) invent room temperatures rather
than returning the room marker. One 64-step client-event run says "The lightning
is purple diamond", which is a literal value echo with the wrong referent and
is not a correct result. Ordinary 7453241/46 also change science answers. No
pilot qualifies; do not increase epochs or corpus scale on this recipe.

Next diagnosis replays a saved approved TRAIN prefix with its supplied future
S-S event and no supplied future word/acoustic code. It compares actual native
first-layer input embeddings with the exact training tensor, then lets native
words/speech codes run freely. Compare base, 64 and 128 on two original TRAIN
records and a paired world. This separates mismatched inputs/deployment from
on-policy trajectory drift; it cannot qualify native turn choice or live tasks.
A literal text diagnostic separately attributes private words to their FIRST
sampled subword, catches guesses that finish after result delivery, and never
promotes matching text to correct claims, actual speech or full-task success.
Two timing/echo regressions pass. The 200ms environment field is also explicitly
scoped to the older Moshi interleaver; Lychee word-offset quality is unqualified.


**Plain words — fixed-state deployment check:** jobs 7453748/49/50 restore
three approved TRAIN histories and supply only the teacher start-speaking
control, with no future word/acoustic tokens. All native input IDs and full
first-layer embeddings equal the actual training tensors exactly (error 0).
Base produces the correct private value in one of three; both 64/128 adapters
produce all three correct short record sentences. This proves the learned
weights can run natively on these exact TRAIN states, not held-out task success.
The next bounded diagnostic activates a checked adapter only for an actual
returned-result response, leaving prior conversation on the base. It requires
exact disabled-adapter baseline replay before any adoption, rebuilds KV at each
switch, and discloses application-controlled response timing. No outcome yet.

**Plain words — owner audio feedback:** the owner reports concurrent capture
7452017 "Sounds smooth". Retain the causal two-GPU recording as the working
audio baseline. This feedback does not approve its factual errors, delegation
behavior, browser/device playback or a microphone product.


**Plain words — result-turn activation resolves this development text failure:**
disabled-adapter 7454004 reproduces every native text, speech and control token
from baseline 7441742 exactly. Cold worker runs 7454007/09 (64) and 7454010/11
(128) all say the correct room-marker sentence for the actual yellow-circle or
purple-diamond world, after delivery with no other colour/shape guesses. Each
uses 20 native packed parameters, preserves sampling RNG, aborts/rebuilds KV and
verifies the refreshed actual engine prefix on both activation and restoration.
This is application-selected reply timing, not native delegation or turn choice.

The initial base reply still incorrectly denies access, and activation waits
14.8–16.4 seconds after result context delivery for the earlier physics reply to
end. Those are unresolved task failures. No spoken-value claim follows from
sampled text. Next is the 64-step checkpoint in actual two-GPU concurrent capture
for both private worlds, with independent transcription of complete utterances.
Interrupted and input-exhausted utterances are recorded exclusions, not passes.


**Plain words — owner resources reviewed and next candidate prepared:**
[The decision note](docs/research/2026-10-15-resource-decisions.md) covers all ten
links, with separate primary-source architecture, alternatives and AEC reviews.
BayLing uses flattened 10:5:10 blocks and zero user-target loss; Moshi uses
summed stream embeddings and time/depth factorization. Current Lychee semantic
training has no user loss to mask. The new assistant-target ownership guard
checks actual sourced text/EOT and causal receipt clocks without modifying
inputs/attention. CPU audit 7454206 passes all 15 examples (170 assistant targets,
zero user/audio/history targets). This is not an AEC effectiveness result.

Concurrent captures 7454091/93 speak both correct private values. Independent
complete-utterance audits 7454208/09 find purple WER 0 and yellow WER 0.1111,
so yellow fails the 0.08 intelligibility bar despite the correctly heard value.
Earlier replies have 4564/3380 actual gap samples; result utterances have zero.
These remain failed full-task/continuity candidates. Baseline 7454087 and audit
7454207 retain zero gaps and exact ASR for both complete utterances; its factual
bread answer remains wrong.

Context Spanning assets job 7454205 fetches public source
79df302fa2820cae10c61bb2b7f302c52edb6ed2 and model revision
e62a4494a295ff8cee345451cf40aae85719a9f5. A bounded native probe is prepared:
strict final full LM load, safe tensor-only checkpoint load, native 80ms steps,
actual worker receipts, span-only injection without a forced response start,
all real input retained, and causal PCM playout. Native retrieval choice and
microphone acceptance are separate future checks. No probe result exists yet.


**Plain words — released Context Spanning works as a late-result component:**
assets 7454205 match pinned publisher checkpoint/voice hashes. Native 7454283/84
(released voice, two values) and 7454285 (accepted synthetic voice) strictly load
16 audio heads with tensor-only deserialization and no unsafe pickle globals.
All 960,000 actual user samples are processed; all three generated texts state
the supplied private value without a forced response start. Native model and PCM
run together at p95 29.7–34.0ms/80ms; fixed 160ms causal playout has zero underruns.
These are scheduled-dispatch/measured-word backend component tests, not native
retrieval qualification or independently transcribed speech acceptance.

All three emit two native retrieval markers, but this first probe services only
one scheduled lookup; follow-up physics is wrong or incomplete. Released-voice
first-frame compute is 2.758/2.814s; accepted synthetic first frame is 0.389s.
Actual span prefill is 105–121ms. Next warm actual encoder/decoder frames before
resetting every stream and conversation seed; then connect sampled retrieval
markers to Whisper recognition of actually arrived PCM on a second isolated GPU
and the actual backend. The new path has no measured/reference transcript and
no scripted dispatch; the backend still chooses the specific tool. Preserve the
speaker-specific target policy before any new model training.


## 2026-10-15 — Context voice accepted; native result use still fails

Plain words: the owner reports Context Spanning's synthetic-reference voice is
“as good or better” than the accepted smooth Lychee recording. This selects a
voice candidate only. Scheduled-dispatch probes 7454283–85 say their actual
private values. Native-RET, actual-ASR and actual-backend probes 7454442–44
(warmed) and 7454577–79 (cold) all recognize the project and obtain the right
actual tool result, but only 7454578 samples the correct value. Other replies
invent/omit it and ordinary physics answers remain unreliable. All speech words,
starts and acoustic tokens are sampled; specific tool choice remains external.

Full-stream independent audits 7454749–58 yield WER respectively 0, .0208, 0,
.0517, 0, .0182, .0667, .0635, 0 and 1.7568. The final synthetic-voice native
run repeats audible material absent from its text; it fails speech behavior.
Other intelligible recordings do not pass factual/delegation gates. Full LM+PCM
runs together at p95 28.5–35ms per 80ms frame. Cold released first-frame stalls
remain roughly 2.8–3s; cold synthetic maximum backlog is about .505s. Warming
full Engine.step removes most cold cost but does not consistently preserve
continuity or behavior: warmed native purple/synthetic captures contain 406/2052
causal-playout underrun samples. Cold native controls have zero underruns.
First cold/warm text divergence precedes actual result insertion; different
nodes/caches/numerical execution remain confounds. No proven RNG-only bug or
quality-preserving warmup fix is claimed.

The next implementation is a bounded eight-update TRAIN-only Context adapter
proof: native 16-head aligned loss, four final temporal layers plus text head,
rank eight with A nonzero/B zero, base frozen, sampling RNG preserved during
construction. Reuse approved clean native caches after every source approval,
fold, artifact, word-timing and receipt check. No arbitrary crop or held-out
failure becomes supervision. User loss targets the clean separate user stem,
retaining genuine overlap; prefix/span targets and invalid IDs are excluded
before CE. This implements an ownership-mask seam, not waveform AEC or measured
echo suppression. No larger run is authorized by component fit alone; require
finite gradients, unchanged frozen hashes, fresh reload and native free-running
retention checks.


**Plain words — training proof guards reject two attempts before updates:**
7455033 stops at RET-marker validation: source/cache lists retain event order,
while token positions are chronological. Keep source-list equality and reject
duplicates, but compare positions against the sorted list; valid cancel scenes
contain e.g. [220,61]. Retry 7455068 revalidates all 48 approved TRAIN scenes and
fails zero-adapter BF16 logit identity before any optimizer update. Compiled
versus replaced gating arithmetic is a candidate explanation, not proven yet.

7455157 adds precise FP32 eager/math checks, a fresh-process reload gate and
native per-head depth comparisons at three frames around a result span.
Pinned Context source retains the older ring cache's final-head mismatch:
streaming excludes the oldest depth slot at q15 while unrestricted batch
training includes it. The batch-only context15 compatibility shim must show
<=.001 FP32 error on every head before optimization; native inference/source
remain unchanged. Keep this implementation correction separate from any
answer-quality claim. The cold adapter reader requires proof/reload hashes and
installs adapters before Engine creates its native CUDA graphs; matched disabled
and trained comparisons use the same saved accepted-reference voice codes.


**Plain words — native proof and actual history comparison pass; behavior does not:**
7455157 completes eight finite updates on four complete TRAIN examples after
revalidating the 48-scene corpus. Rank-eight adapters in last four temporal
output/FFN layers and text head contain 1,894,400 parameters; raw QKV/depth/codec
parameters stay frozen. Base hashes are unchanged. Peak allocated/reserved are
34,061,415,936 / 43,130,028,032 bytes; measured training/checking section is
31.90s. Fresh process exactly reproduces all valid FP32 text/audio logits.
Checkpoint SHA d2086be61657383141510a234ae5f3b51325ffb315d7d7b048ded2845c53455b.
FP32 native depth comparisons show old head15 errors .07553/.07886/.19199;
batch-only context15 gives all-head maximum error <=9.54e-6.

7455458 actually captures released prefix/span/live model input IDs. Source
startup omits first prefix C0; after correcting that oracle, ten observed
acoustic entries still differ near the span. Candidate prefix bootstrap plus
restoration only of pending explicitly provided delay1 codes makes all 5,372
IDs (17x316) exact. Generated/unprovided assistant acoustics are never copied.
This is a single-span forced-TRAIN layout check, not full graph/logit parity or
production speech acceptance; actual free-running provided-user traces follow.

7455516/17/18/19 are released/layout-only/zero-wrapper/eight-update yellow;
7455520/21/22/23 are their purple counterparts. All process 960,000 real samples,
have zero fixed-160ms simulated-playout underruns, p95 .0288–.0341s per 80ms,
and cold backlog 2.81–3.07s. They remain task failures: released yellow says the
value alongside invented prices; layout-only yellow omits the shape; eight-step
yellow invents $179; eight-step purple states its actual value but gives a bad
physics explanation. Eight updates are a feasibility proof, not learned-task
success. Zero-purple7455522 executes the correct private read but backend
content is empty; mark that delivery invalid instead of assigning the value
omission to the foreground. New worker-contract regression rejects empty text.

Speech audits 7455646–53 report WER respectively .13699/.01639/.11667/.02041/
0/.04054/.01449/.01471. Numeric formatting partly explains money mismatches;
7455648 also fails ASR time/word bounds and has unfinished trailing speech.
No full-task or real-microphone claim follows. Bounded pilot7455712 starts
fresh from released weights (not an eight-step warm start) and balances all48
TRAIN scenes for128 updates. Native proof, corpus/voice hashes and layout audit
are required gates; factual and speech quality remain separate forthcoming
comparisons. Thirty local worker/ASR/ownership/playout regressions pass.


## 2026-10-15 — Reject broad pilot; isolate native padding and request policy

Plain words:128 updates do not qualify the Context actor.7455712 trains all48
scenes in52.08s with34.06GB peak allocated, frozen hashes unchanged and exact
fresh FP32 reload. SHA12635510464a68aeceea7ab2826da10b33750377306b6ca68bf794941d3918b2.
Native7455763–76 compares private000/001 paired worlds and simple004/005,
clarify006,cancel007 with zero-wrapper controls. Six of eight pilot cases have
zero or one sampled word and no RET. Pencil005 improves in isolation; cancel
still fails. Thirteen independent audits7455873–85 record the quiet/failed
outputs; Whisper's “You” on silence is not trustworthy audible speech evidence.
Zero clarification7455773 tries a private read before user identification and
is correctly blocked. New native error handling retains the failed verdict,
receipts and diagnostics, supplies a nonspoken failure/clarification notice and
continues input/PCM. Thirty-three local relevant regressions pass.

Teacher calibration7455921 uses only48 TRAIN histories. Word NLL1.887→1.299,
top1 .549→.666; private-phrase NLL .439→.430/top1 .789→.779. RET target NLL
20.06→7.87 but remains0/40top1. Its probability on ordinary word slots falls
.0413→.00178. The old report key `eot` is a diagnostic naming mistake: ID0 is
**end-of-padding**, not end of utterance. Existing Interleaver adds0 before
speech following padding; native Context preparation uses3 everywhere else.
Actual ordinary-word first frames match floor(ASR onset*12.5), except when
queued collisions delay them. Do not invent a fixed80ms shift.

Native codec7456232 compares the same approved20.08s PCM across old batch2,
Context batch2, Context offline batch1 and Context streaming batch1. Both batch2
speaker outputs exactly match cache251x8 IDs. Context offline user differs in
only two residual-code entries; streaming differs per head[3,12,14,17,17,15,16,5],
with zero shift best and first any-head mismatch at9.44s. Semantic agreement is
98.8%; this is one TRAIN spot-check, not complete streaming identity. Causal
codec history/precision remains a separate long-context check.7456078 initially
failed on omitted Triton driver linkage; fixed6232 performs actual measurement.

Padding-only preflight7456094 maps0→3 without changing words/audio/RET frames
or result clocks. All5,372 forced native input IDs match its clean oracle, and
eight finite updates plus exact fresh reload pass. SHA63f7d31c0e31eed5de5d1173607263142ea3bb893a84d49468e822211cc3f8c0.
Its128-step bounded ablation7456282 starts from released weights and is submitted;
no improvement is claimed. The new source explicitly records the changed padding
convention. RET policy is still the old actual/declared dispatch annotation,
not the native paper's pre-lead marker. Any later change must preserve actual
source clocks and label new policy targets as derived, with user information
and receipt causality checked. Unqualified development failures remain excluded
from supervision. Pending-state-only7455886/87 and recovery7455888/89 fail
behavior; do not scale prompt variants or the rejected128 checkpoint.


## 2026-10-15 — Padding ablation and temporal computation check

Plain words: padding-only128 pilot7456282 passes the technical training gates,
not behavioral qualification. It sees all48 TRAIN scenes in52.49s, with unchanged
frozen hashes and exact fresh-process FP32 reload. Checkpoint SHA
`3e9d4011a23858e505878c43a7b2ffbf8af4cda89480ad4776ecd660d006c13c`;
data SHA75c8a6dee846420b3319984fc40b041240d0ca109bbda616f967bef3568e5796.
Native tests7456656–63 use the same saved voice and compare private000/001 paired
values plus simple004/005, clarification006 and cancellation007. Existing
zero-wrapper controls remain separate. A new optional canonical FP32/math
history check compares streamed hidden states and text logits with the complete
batched input, beyond the already measured exact token IDs. It does not change
training or inference behavior and does not establish production graph parity.
Owner voice feedback for7454285 is recorded on the listening page.


Plain words: native7456656–63 all sample empty text and zero RET. Reject the
padding-only pilot for scale. Temporal audit7456824 passes FP32/math with all
17x316 input IDs exact, maximum hidden error.0003051758, maximum text-logit
error.0000988245 and zero top1 disagreements. This is a forced TRAIN component
through one span, not a production graph/free-running qualification.
The next conversion derives native pre-lead controls separately from source
worker clocks, preserving every word/audio/arrival target. It requires arrived
user identification, complete cancellation instruction, no collision and no
receipt preceding its request. Validation/test labels remain forbidden.


Plain words: native-policy preflight7457079 accepts all48 TRAIN conversions and
completes eight finite updates with frozen weights unchanged; retrieval of its
fresh reload is pending. A separate actual-history collector records graph
inputs outside graph capture and direct prefix/span inputs, without sampling
changes or future labels. Only explicit approved TRAIN scenes are permitted.
The recorded history is unapproved conditioning; a future assembler must mask
every past label and independently check its new continuation. No failed
validation case is reused for this collection.


## 2026-10-16 — Native request labels; actual student histories; access block

Plain words: all48 TRAIN scenes pass the native-prelead conversion. All40 RET
labels move.16–.80s later (median.44s), without changing source dispatch clocks,
words, audio or actual result arrival times. Preflight7457079 completes eight
finite updates and exact fresh FP32 reload; SHA
`0b20cc01db2174fb114201f81e41cdf0ddeb5373d3f72edfc3efca898ba59716`,
data SHA7c0a3c6f42675766b0a3edb4569dcad1ad40a8619a21215d11330107d9926aef.
Integrated history/temporal checks pass with zero text-top1 disagreements.

Native7457163–70 all sample nonempty text and make1–2 requests, but four private cases omit
their requested values.001 consults the backend for physics after missing the
record request. Clarify006 guesses an unidentified project (blocked by the
worker guard) and says M room. Cancel007 emits no later request to cancel.
Simple005 correctly explains graphite;004 adds an E F T preface before a broadly
correct dissolution answer. All real input is retained; only cancel007 has10
samples of simulated-playout underrun. No fresh holdout or microphone pass is
claimed; independent new speech audits have not yet run.

Raw conditioning collectors7457161/62 complete approved TRAIN000/008, with
1x17x780/626 delayed input IDs and501/355 actual graph calls, plus direct
prefix and2/1 span forwards. Metadata's contiguous call/frame accounting and
source code pins are checked locally. Both have zero playout underruns. Raw
`.pt` file hash/load verification and independently checked future targets remain
required; neither recording is supervised data. Thirty relevant local causal,
worker, ASR and playout tests pass in `.venv-api`; system Python lacks NumPy.

The latest bastion trace explicitly says “Not allowed at this time” before
authentication. No native-policy128/calibration submission returned a job ID.
Automatic connection recovery stops pending the owner's manual access check,
in accordance with the official Isambard agent access guidance. Certificate
expiry remains16 October10:39:50 Berlin. Next batch is concretely prepared in
`evals/context-native-policy-next.sh`; syntax/hash constants are locally checked,
but this helper has not executed remotely. It gates a bounded pilot and its
eight dependent behavior replays on the saved, verified preflight.


## 2026-10-16 — Access restored; PAD3 calibration and observed EPAD ordering

Plain words: renewed Clifton works over normal SSH at07:46UTC. Certificate
expires21:46:11Berlin. The prepared batch submits pilot7463729, calibration
7463730 and eight native dependents7463731/33/34/35/36/37/38/39. Pilot reload
passes; SHA7a6894b3604209188e0446b7df68686e2d0d99ed7d2f5a79d77b5a6cd9c1e51f.
No behavioral improvement is claimed before those tests finish.

Calibration7463730 uses all48 PAD3/native-prelead TRAIN histories. Zero→8-step
word NLL11.339→11.206, private NLL10.426→10.322, RET NLL20.545→20.499/top1
0/40. This differs from the old legacy-label diagnostic, so do not call it
regression on identical inputs. Direct free-running accepted captures7454283/85
contain29/46 EPAD0 tokens, including0→4→first-word at both native RET events.
The next explicit variant retains those cues, repositions the lead cue one
empty frame before derived RET and preserves other EPAD/word/audio targets.
Seven pure causal/conversion regressions pass; full-corpus/GPU checks follow.


Plain words: reject pilot7463729 for scale. Native7463731/33/34/35/36/37/38/39
has seven empty sampled-text streams and zero RET in all eight cases; pencil005
alone produces an unqualified answer. No new speech-audit pass is claimed.
Raw-history validator3bf031c passes saved7457161/62 tensor SHA/load, immutable
collector/voice/model/source pins and complete real/virtual frame accounting.
Both remain conditioning only, not approved training targets. EPAD-retaining
preflight7463825 is now submitted from immutable44b239e.


Plain words: paired raw TRAIN collectors7463841–48 cover000–003 in two approved
teacher worlds each (original plus one checked counterfactual). They remain
unapproved conditioning. A v2 recorder is prepared to save small generator
cache/provided/offset and RNG snapshots after actual complete steps, retaining
0 provided pending assistant acoustics and7 actually observed user residuals.
This will allow native forced-gold continuation assembly without fabricating
pending assistant audio. No transformer KV or model weights are saved. The
new snapshot path still needs an actual GPU collector check.


## 2026-10-16 — EPAD preflight and actual-state restart gate

Plain words: EPAD-retaining native-prelead preflight7463825 passes eight finite
updates, native history/parity and exact fresh reload. Adapter SHA
8424628a0adabcab6c5ed7147e478b0023f9c589455d348b04d77b9cce8cc279.
Calibration7463976 gives baseline RET top1 36/40 and word NLL1.897, compared
with 0/40 and11.339 on the earlier PAD3 convention; this is a different input
convention, not an unchanged-input model improvement. Eight updates barely
change these teacher-history measures. Free-running7463977–84 remain decisive.

Plain words: v2 actual TRAIN collectors7463985/86 finish with small generator
state and CPU/CUDA RNG snapshots, zero provided pending assistant acoustics,
and seven provided pending user residuals. They are unapproved conditioning.
The next diagnostic replays actual consumed direct/graph calls, restores cache
ownership and RNG, and requires two unforced following steps to match the
recorded snapshots. It creates no gold targets or optimizer updates and makes
no PCM/microphone restart claim. See the dated actual-context source review.


Plain words: restart7464207 fails its strict gate. Both recorded TRAIN histories
pass raw integrity and replay clock/storage checks. Two next frames match all
provided masks, offsets, RNG and delayed returned text. Assistant acoustic cache
IDs diverge: orange is exact at step1 then2 residuals differ; purple differs
from step1. No histories are approved for gold targets. Ranked hypotheses:
(1) warm graph path differs from original cold capture, (2) replay numerical
differences alter sampled acoustics, (3) unrecorded depth state. Next experiment
changes only graph warmup; acceptance remains complete next-state equality.


Plain words: cold-graph diagnostic7464272 also fails assistant-acoustic exactness,
while text/ownership/clocks/global RNG still match. Warmup alone is not the
fix. Prepare v3 collectors saving each actual native call's final hidden vector
and text logits outside the graph. Replay will compare these before sampling
to separate temporal numerical differences from depth/sampler state. Raw
conditioning and all future targets remain unapproved; thresholds are unchanged.


Plain words: EPAD128 pilot7464214 passes fresh reload but native7464223–30
again fails behavior: four empty sampled-text streams (both001, simple004,
clarify006);000 answers are missing/wrong-field, cancel007 falsely claims room
marker ready. Reject for scale. Eight-step7463977/78 spoken yellow circle and
purple diamond are independently confirmed by audits7464215/16 (WER0/.026).
Other speech audits confirm failed001/clarify/cancel behavior; cancellation
transcription has invalid bounds and is not an approved artifact. No overall
model qualification or PCM-silence conclusion is inferred from empty text.


Plain words: v3 collectors7464403/04 finish with actual numerical evidence.
Restart7464405 stops on the explicit next-input equality check after sampled
acoustics diverge, before writing its numerical report. Keep the same strict
requirement but record this failed check in the report so the first numerical
difference survives. No optimizer or target approval is involved.


Plain words: numerical restart7464529 locates differences before sampling:
all381 compared calls differ across both histories, including the first persona
prefix (hidden max0.25, text max0.1875). The first following input is exact;
the second differs after sampled audio divergence. RNG is downstream of the
first observed mismatch. Source/execution settings need explicit recording.
Cold numerical replay is submitted separately; no approval threshold changes.


Plain words: cold numerical7464626 still differs from the first prefix, ruling
out replay warmup alone. Same-label TRAIN calibration7464552 shows128 steps
fit ordinary words (NLL1.897→1.308) while native behavior worsens; this supports
the teacher/student-history mismatch diagnosis, not scaling dense training.
Next single-variable numerical control selects Torch math attention for both
actual capture and replay. The native default is preserved; weights, user audio
and sampling settings are unchanged. Execution flags are now explicitly saved.


Plain words: helpers for future-target selection now enforce identical TRAIN
user PCM/word timing (only zero-tail padding may differ), an actually delivered
successful matching private receipt, and a complete independently measured
source sentence after its teacher receipt. Wrong worlds, failed notices,
cancelled tasks, early targets and token collisions are rejected. Three local
causality tests pass. These helpers do not approve labels, write a training
dataset or bypass the still-failing strict restart gate.


Plain words: execution-provenance7464633/34 records identical model dtypes,
Torch/CUDA/GPU and attention/TF32 settings, yet prefix outputs differ. Matched
math-attention control7464640/41 also fails (prefix hidden max0.5). Neither
warmup nor attention selection alone explains it. Next capture records cloned
per-embedding and per-layer prefix outputs to find the first numerical
divergence. Hooks run only on the direct prefix; source/default model behavior
and the strict restart requirement are preserved.


Plain words: prefix trace7464778/replay7464781 matches all saved last-column
embeddings, transformer input and layer0. The first saved difference is layer1
(max0.00048828125), growing to output norm0.25. Earlier columns are not yet
traced, so this does not prove all layer0 values match. Next replay changes
only process-global kernel priming: reproduce Engine constructor's four sine
steps and eight-column span before creating the fresh conversation stream.
Native stream reset does not clear lazy compiled closures. This is a tested
hypothesis only when the next gate returns; no root cause is claimed yet.


Plain words: replay loader currently re-encodes the accepted reference before
creating LMGen. Actual zero-wrapper captures load verified voice.pt and skip
that encoding. Mimi also calls shared lazy RoPE code, so equal resulting voice
IDs do not imply equal process compilation history. Add an explicit recorded-
voice diagnostic that verifies its captured hash and reuses the exact codes;
ordinary training defaults remain unchanged. Test it separately after the
native-constructor priming control, with all source histories immutable.


Plain words: recorded-voice controls7464929/31 now reproduce the entire saved
last-column prefix trace exactly, with/without constructor priming. Subsequent
acoustics still diverge. Full-engine diagnostic now replays the same approved
FLOAT user PCM, both native codecs and recorded real-frame span boundaries,
without new backend/ASR calls. The serialized user_in.wav is PCM16 and must not
replace the original FLOAT input. No new target labels or optimizer updates
are created, and replayed dispatch is not a live routing qualification.


Plain words: full-engine7465002 fails first cache equality at real frame64;
this first variant retained the trainer's disabled codec CuDNN flag, so it is
not a matched native execution test. Corrected7465018 explicitly restores
source codec enabled/deterministic/benchmark flags before Engine construction.
Its result is pending retrieval. A single latest SSH read hit a banner timeout,
not an authentication rejection; normal retry follows, without access changes.


Plain words: normal SSH retry succeeds; the banner timeout was transient.
Corrected full Engine7465018 still fails (first cache divergence frame64);
first temporal difference is call6 while actual input IDs remain exact until
call66. Captured/default attention numerical replay is not yet exact even
with codecs restored. Next control uses existing math-attention source7464640
with full Engine and recorded voice, combining the two separately identified
requirements. It does not require new speech generation or backend calls.


Plain words: full Engine MATH replay7465797 PASSES all189 real frames, all
captured input columns, temporal hidden/text outputs, cache/provided values,
clocks and CPU/CUDA RNG. No thresholds relaxed. Required combination: recorded
voice, actual codec operation order, matching math attention. This is a bounded
actual-history component, not live backend/microphone/complete-model acceptance.

Plain words: prepare approved same-user/same-world teacher sentences after a
quiet actual failed-reply history. Replay must pass at that selected cut too.
Only then force independently approved future text/assistant codes through
native LMGen, verify every future user codec against the original stream, and
record current hidden/text targets. Pending pre-cut residuals stay unprovided
and sampled. All past labels remain absent. This initial component has no
audio loss, no optimizer and no voice/task qualification. Four more math TRAIN
collectors7465804–07 are submitted in approved worlds000–003.


Plain words: CPU eligibility finds three corrective candidates:7465804 (cut324,
32 future frames, orange square),7465806 (cut221,56frames,purple triangle),
and7465807 (cut256,35frames,orange triangle).7465805 is excluded because its
correct value was already spoken; do not teach unnecessary repetition. Native
dependency overlay is required for SentencePiece even in CPU-only checks.
Quiet selection now derives packets from hash-verified model.wav and requires
complete ledger/gap/sample/clock accounting; it does not trust an unhashed
packet-stream side file. Eleven related local guards pass. GPU cut/replay and
forced-future ownership validation follow; no optimizer has used these plans.

## 17 October 2026 — rejected duplicate-cue boundary experiment

Plain words: new response-boundary source05d7b28 passed CPU plans7502645,
native captures and speech for three replies7502654/55/56, mathematical
checks7502672 and eight-update reload7502676. It is nevertheless REJECTED:
the returned text lags prediction by one frame, so it repeats an EPAD control
already present in the actual cache. No continuation is submitted. Saved
native states prove this exact delay in three TRAIN traces. Corrected
source787410a directly substitutes the first answer word for the RET prediction;
selectorbd70fc8 rejects old versions and changed planner hashes. CPU7502998
passes four corrected plans and four legacy regressions; all25 guard tests
pass, including regressions that failed before these fixes. Native capture,
speech and mathematical checks are still required. This fixes the new
composition's clock; it does not establish a cause or cure for all earlier
conversation failures.

## 17 October2026 — corrected decision timing, controls and routing tradeoff

Plain words: corrected nineteen-component7503114/7503303 passes numerical and
exact resume checks but remains unqualified on native3369–78 and corrected
TRAIN3464–69. TRAIN3469 executes a real cancellation then speaks a false denial;
speech3524 confirms it. Explicit read-only trained replay3670 reproduces the
actual state without labels or new backend calls. First-word-weight16 proof3477
and136-stage3528 change three ordinary TRAIN requests into direct answers, but
private0003557/changed-world3565 both invent7.22 without lookup. Reject unchanged
expansion. Constant RET-logit biases-2/-4 on reference7500125 also fail3275–80;
stronger bias suppresses required private lookups. Next supervision must balance
positive lookup decisions with ordinary answers. Existing complete private
futures contain a later backend-answer dependency; extract only an independently
verified pre-result lookup prefix, not the full answer-bearing tail.


## 17 October2026 — positive lookup prefixes and actual-query anchoring

Plain words:CPU7504055–57 derives exactly one lookup label per verified prefix.
Audit7504247, eight-update7504403 and136-stage7504434 pass mathematical and exact
reload checks. Native4494–4509 restores lookup but regresses direct ordinary
answers and still fails full conversation tasks. Reject unchanged expansion.
Actual-ASR query anchoring2b8d6f5 in4409/13,4411/15,4412/16,4425/26 also fails;
context labels become audible and results are contradicted. Independent speech
7504102 confirms both ordinary component gains and audible factual failures of
weighted19. No model qualified; cloud remains paused.


## 17 October2026 — trained-state recovery and bounded learning curves

Plain words:retention pairing is corrected in d989e20; actual136-stage4434 had
each of six kinds paired with only two of four references. Matched136/520
checkpoints4805/4806 restore exactly.520 learns three ordinary first-word
choices and directly answers evaporation/day-night in sampled native text, but
private, clarification and cancellation still fail.20-data acknowledgment
variants4893/4896 also restore and fit their targets but regress native lookup.
No unchanged extension. New81b9269 plans corrections on the actual520 model's
TRAIN states: CPU5110 passes eight plans, four legacy regressions and eleven
cluster tests; eight native replays/future captures5207/5214/5216/5217/5218/5219/
5220/5221 all pass. Speech5283 passes the five own-speech early captures at the
unchanged0.15 bar.26-data mathematical audit5317 and dependent eight-update5318
are submitted; no new model qualification is claimed. Cloud stays paused.

## 2026-10-17 — Trained control transitions and reasoning separation

Plain words:26-data136/520 models5421/22 pass reload but their32 native development
runs5474–5505 fail full qualification. Correct first clarification is followed
by no lookup; returned values are omitted/contradicted and cancellation fails.
Independent ASR5897 confirms several observed words, including correct first
questions and an incorrect lightning explanation. Capability-only cancellation
5528 executes a real cancel on4806 but does not repair the full conversation.

Plain words:source8a1c201 passes37 local guards. Exact-prefix native captures5893
(clarification lookup) and5895(cancellation), CPU plans5869/70 and independent
CPU audits5894/96 pass. Three expert controls are rolled forward; fourth frame
is an unlabelled native readout, no tool dispatch or speech/fact supervision.
28-data audit5910 and dependent warm eight-update5911 are submitted, same4806
parent and recipe as26-data. Qualification remains pending. Zero-adapter native
ordinary5883–88 all delegate and several give poor answers; publisher-persona
5899–5906 checks whether the records-focused persona contributes. Reused scenes
are development diagnostics, not fresh general intelligence evidence.

## 2026-10-17 — 28-data native verdict and teacher-retention comparison

Plain words:28-data audit5910, eight-update5911 and520-update5957 pass exact
reload. Native5961/5964–5978 (16 total; use receipt, not arithmetic) still fail
full-system qualification. Two real component gains: TRAIN0205968 executes
cancel_lookup after complete cancel audio; validation0065975 clarifies then
executes its lookup. Neither uses the subsequent result/status correctly.
Ordinary answers and private results remain unreliable. ASR5984 covers19 WAVs;
invalid word timing on5968/5976/5906 blocks spoken acceptance, and5977's Copper
Dive does not verify the expected copper diamond. Do not extend5957 unchanged.

Plain words:sourceb0f1beb introduces explicitly audited teacher-text CE on the
existing four retention examples, alongside unchanged base KL/hidden MSE.
CPU6263 passes scope and real Torch gradient/mask/state tests. Default teacher
weight0 remains unchanged. Weight1 comparison uses identical28 components,
parent4806, schedule and8+512 budget:6461proof,6462fit,6463–6478native,6479speech.
It tests whether preserving a poor base distribution without teaching these
approved complete teacher words contributes to failure. It does not establish
that diagnosis or change the backbone. All data stays TRAIN-only.


Plain words:17 October20:28UTC — teacher-retention6462 completes and exact reload
passes. All16 native development runs6463–6478 plus independent speech6479
complete. Reject: excessive repetition, wrong answers, missing validation
clarification6475; actual TRAIN cancellation remains a partial control success.
Speech6467/6475/6476 has invalid word bounds, so no spoken acceptance there.
Checkpoint/state and16 native WAVs are locally hash-verified. Sourceb42a6e4's
next comparison changes only future acoustic conditioning for three approved
late TRAIN private targets. CPU6900 passes3plans,7legacy regressions and13Torch
checks. Captures6926/6930/6932, CPUaudit6934 and speech6936 submitted; no optimizer.


Plain words:17 October — matched own-audio private model6978 PASSES520 updates
and exact fresh reload. Completed unforced development runs already contradict
full qualification: wrong lightning/day explanations, missed lookup after project
clarification6983, contradictory speech despite actual cancellation6984, false
physics after yellow-circle lookup6985. Do not extend this unchanged. Remaining
native runs/independent speech6995 remain pending. Metadata inventory establishes
25 existing GPT-Live TRAIN teachers (19private/4simple/1clarify/1cancel); the
failed teacher-text recipe supervised four longest only. New all-teacher CPU
scope audit is distinct from optimizer approval and retains all fail-closed masks.


## 2026-10-17 — Native teacher input audit and observed source boundary

Plain words: actual Engine inputs7800/7801/7852/7803/7804 differ from the offline
cache only on user acoustic rows. Independent CPU8370 passes exact all17-row
reconstruction using actual streamed user codes; assistant targets/masks unchanged.
Broader captures8408–8427 expose one phantom end frame in CF0062 (8420); CPU8428
and dependent math8559 cannot pass. Source900480 samples/24000 is469 physical
frames, but floating37.52*12.5 rounds above469 and pinned interleaver usesceil.
All16 acoustic rows of nonexistent frame469 are-1; no word/control/span is lost.
Source4b0f9d5 adds strict opt-in observed-boundary scope, retains originals and
binds historical captures by immutable source and explicit scope SHA. CPU8779
submitted;12 real Torch guards pass so far. No optimizer/quality gain claimed.


## 2026-10-18 — Full native teacher audit and numerical reference

Plain words:CPU8779 passes observed physical boundaries/all25 scopes; CPU8795
passes all25 actual Engine input reconstructions. Text/assistant rows/masks
unchanged, user encoding explicitly streamed. FP32math8796 fails2/25 at fixed
0.001:clarification0190.001466, CF00410.002780; all causal errors0. Cancelled
blocked proofs8811/12. Zero-update9019 repeats the clarification failure exactly;
FP64attention alone does not pass, but9034 FP64linear-only max0.000253 does,
with identical supervised CE and no eligible top1 changes. Sourcef16d430 adds
separate v2 reference verification retaining original failed FP32 results; no
trainer/runtime precision change. Two real Torch isolation guards pass.
Reference9062 running; dependent matched8-update proofs9077/78 and520-step
models9079/80 queued. All25 native teachers remain TRAIN-only; no quality claim.


## 2026-10-18 — Matched teacher coverage rejected; preservation-loss ablation

Plain words:precise-reference9062 PASSES all25 at unchanged0.001 (max0.000883),
retaining original FP32 failure8796. Proofs9077/78 and520-models9079/80 PASS exact
reload. All32 unforced development cases and ASR9126/9143 complete; both remain
UNQUALIFIED.9080's audible false lightning claim, no TRAINcancel action, omitted
orange-circle receipt and invented258-dollar cost contradict complete task use.
TRAIN9131 improves clarification+lookup+browntriangle+evaporation, but validation
still fails.16/16 ASR timing bounds valid for each arm; causal playout underruns
occur10/16 four-teacher and6/16 all25. No microphone/fresh-holdout acceptance.
All model/state/voice/data and32 WAV hashes verified in local backup.

Plain words:source60aa2c9 introduces an explicit0/1 base-preservation multiplier.
Removing it requires independently approved native teacher supervision; default
calculation unchanged. It tests whether preserving the unreliable base resists
correction; exposure mismatch and narrow supervision remain alternate hypotheses.
Same28 data/4806 parent/all25 teachers/weight1/budget; only base KL and hidden-MSE
weights become0.11 local guards pass. Proof9708 PASSES8/exactreload and state
restore, four local hashes pass.520-fit9710 running,16 native cases9762/9765–79
and independent ASR9780 gated. All in workq; bounded interactive wrapper1b5c854
was prepared/checksummed but unused because original proof started normally.


## 2026-10-18 — Spoken teacher supervision comparison prepared

Plain words:base-retention-off9710 completes520 updates and exact fresh-process
reload/optimizer-RNG restoration; checkpoint/state/data/voice hashes match local
backup. All16 unforced native cases9762/9765–9779 complete. Model text has a
correct lightning explanation9762 and more retrieved values, but collapses into
repeated "cancelled" on several cases and only "because" on both validation
ordinary questions9774/9775. ASR9780 remains pending; these are generated-text
observations until independent recognition. No qualification or unchanged extension.

Plain words:the approved25 TRAIN teachers contain993 spoken text targets among
11,425 eligible real-frame text targets (8.69%). This is measured coverage, not
proof of the cause. New opt-in balanced-spoken native teacher loss assigns half
the teacher objective to spoken tokens>=5, half to clock/control tokens<5, retaining
PAD3 weight0.5 within its group. Every original label, mask and input is unchanged;
no prefix/span/user/audio loss. Default legacy loss is unchanged. Policy/source
hash enters portable optimizer metadata; each step records both group CEs/counts
and legacy versus balanced spoken weight mass. Complete native CPU/math approval
and all25 coverage are required. Proposed matched arm retains original base KL+
10xhidden preservation, same28 data/4806 parent, teacherweight1 and8+512 updates,
compared with9080. It tests signal dilution without adopting unstable9710.
Seven local pure-policy guards and Python/shell syntax pass; three real Torch
gradient/ownership guards must pass in the cluster proof before any model update.
Teacher-forced versus own-speech history and narrow1clarify/1cancel supervision
remain competing explanations. No claimed reasoning or microphone improvement.

Plain words:next zero-update diagnostic compares complete teacher words with
native sampled agent audio, on exactly four existing TRAIN histories (one per
kind). New explicit teacher-engine-native-audio mode supplies only text/user
tokens, verifies all agent provided flags false, independently reconstructs all
non-agent native rows, and records actual generated codes/audio and full inputs.
It retains the measured teacher word timing and recorded context arrivals, not
live backend results. Its distinct schema/files cannot enter the existing
teacher-input approval path. Independent speech audit labels forced words and
requires TRAIN provenance, matching WAV hash, full generated frame coverage and
input/ownership checks. No optimizer or data approval follows from rendering.
Existing two integer-input Torch guards plus a new forcing/ownership guard run
inside each cluster capture before loading the model. Local syntax passes;
local Torch execution is unavailable and is not reported as passing.

Plain words:render7510411 fails v1 ownership after all3 Torch guards pass; dependent
413/414/415 and speech416 cancel without running. All9 non-agent input rows are
exact, but seven delay-one agent rows have provided=True at real frame0 only.
Independent CPU inspection reproduces their exact values from the pinned persona
prefix:243/1178/546/1736/1030/1978/2008. Every other agent provided flag is false;
text/user flags all true. This is pending persona-prefill state, not supplied
teacher audio. V1 incorrectly required every agent flag false. Preserve its failed
report and trace. V2 requires exactly those seven initial flags, exact prefix code
values, all other agent flags false, and unchanged all9 non-agent input equality.
Negative guards reject a changed prefix code, unexpected later provided flag,
missing user ownership or changed teacher text. Runtime/training layout is unchanged.
New captures require new immutable source; the old failed run is never relabelled.


## 2026-10-18 — Overnight matched loss and complete native speech-history verdicts

Plain words:balanced-spoken sourceca66cda proof7510330 and520-continuation7510331
PASS exact fresh-process reload/optimizer-RNG restoration; both model/state/data/
voice hashes pass local backup. All16 unforced cases7510338/7510340–54 and ASR355
complete. Reject full qualification: triangle342 audibly says3,4,or5 (ASRworderror0),
pencil350 audibly claims lead is made of lead (worderror0), both cancellation
scenes lack executed cancellation, clarification343 is repetitive with invalid
ASR segment/word bounds.15/16 have valid bounds. Ordinary trained lightning338,
evaporation340 and day/night341 improve, but private/control/factual reliability
still fails. Do not extend unchanged or promote from teacher loss. All16 WAV
hashes match local reports; the48-player overnight comparison now includes this arm.

Plain words:sampled-acoustic teacher probes onf306015 all PASS structural v2
ownership/non-agent input checks:498clarify/503private/505triangle/506cancel.
Independent speech507 completes4; worderrors0.10/0.086/2.0/0.44, all timing bounds
valid. Triangle teacher text3 is recognized asThank you; cancellation adds a
badge-phrase fragment outside supplied words. ASR correspondence is diagnostic,
not human listening. Do not expand these captures into training. Full teacher
text forcing at recorded timing does not ensure speech correspondence. Original
v1 failed411 is preserved, with exact prefix-state diagnosis; its dependents never ran.

Plain words:isolated CPU TTS compatibility584 completes onARM64/Torch2.9/CUDA13.
Actual Moshi0.2.11 TTSModel/CheckpointInfo imports pass from the SHA-pinned wheel;
shared environment untouched, no model weights/GPU. Several installed versions
are outside upstream bounds (Torch2.9, safetensors0.6.2, HFhub0.34.4,
sentencepiece0.2.1,sphn0.2.1); imports are not supported-version or GPU proof.
Next bounded step retrieves exact pinned public TTS assets and a published voice,
checks published LFS hashes and every extracted wheel byte, then may run a
single-GH200 compatibility/speech trial. Native voice identity does not transfer;
this is a separately disclosed stronger-controller/streaming-speech hybrid path.
Cloud remains paused. A transient SSH relay banner timeout recovered on retry;
current certificate expiry18Oct11:39Berlin, so no overnight user input was needed.

Plain words:asset retrieval7510636 PASSES in35seconds: exact HF model/voice
revisions, seven assets, published weight/voice SHA256s and sizes, and clean pinned
upstream DSM Git source4c4f65e. TTS weight3,683,719,712bytes SHA726ddadd90a080c8...
matches publisher; existing Mimi384,644,900bytes/09b782f... is byte-identical and
reused. Every isolated package byte matches the reviewed wheel. Bounded GPU probe
is prepared: strict complete model/Mimi loads, one actual gpt-6-luna lookup over
RecordWorld, and exact returned text rendered through upstreamTTSGen with published
speaker embedding. It records actual PCM availability, causal playout, raw packet
stream, loading/runtime versions and backend receipts. Typed input and complete
controller text before rendering are explicit; no concurrent listening, microphone,
native request decision, native intelligence or full-system acceptance is claimed.
Independent recognition has a separate external-controller source kind. Syntax,
CLI parsing and shell checks pass; actual GPU/speech checks remain to run.

Plain words:first GPU TTS679 fails in21seconds before speech/backend execution;
ASR680 cancels. Safetensors0.6.2 rejects32 legacy Mimi projection names before
the registered official Moshi attention conversion hook runs. Existing required
package bound is safetensors<0.6. Ranked predictions: raw-name prevalidation conflict
implies ordinary strict PyTorch loading passes; wrong checkpoint architecture
implies that also fails; further GPU/runtime incompatibility remains possible
after loading. Do not downgrade the shared environment or skip weights.
New CPU gate must reproduce the original failure, load through the official hook
using load_file+load_state_dict(strict=True), and independently compare every
parameter/buffer with the immutable source (all32 legacy projection mappings,
no unused source tensors). Negative controls alter a value, remove a tensor and
add an unused tensor; all must fail. A temporary Mimi-only loader restores on exit,
changes no package/weight bytes, and requires that CPU gate before a new GPU probe.
Native models remain rejected and unchanged.
[Safetensors0.6 source](https://github.com/huggingface/safetensors/blob/v0.6.2/bindings/python/py_src/safetensors/torch.py).

Plain words:CPU722 PASSES actual original failure reproduction, strict official-hook
loading, exact318 Mimi parameters/buffers and all3 integrity negatives. NewGPU723
PASSES complete strict Mimi/TTS loading and actual backend lookup. Its sentence
"The room marker for Project Willow North is a teal hexagon" is independently
recognized by724 at worderror0 with valid timing. This is typed external-controller
content, not native reasoning/duplex acceptance. Eager execution is NOT realtime:
4.8s audio takes20.823s after request, backend3.021s, firstPCM7.977s,
framep950.233s and58 causal underruns. Keep raw packet stream separate from causal
playout. Next matched renderer benchmark holds the exact prior actual backend
text/receipt fixed, discloses receipt replay, and compares warmed eager versus
warmed CUDA graphs with Torch compilation off. No new backend execution is claimed
for replay arms. Report cold preparation/warmup separately, require both main/depth
graphs really captured, and retain per-frame/decoder timing plus independent ASR.
Predictions: depth kernel-launch overhead should fall with graphs; a decoder
bottleneck would remain visible; backend delay is isolated by the common replay.

Plain words:warmed eager779 and CUDA-graph781 both PASS the same replayed actual
controller sentence; independent780/782 worderror0 with valid timing. Both use
the same pinned weights/voice/text/seed, Torch compilation off and MATH attention.
Eager framep950.206s, decoderp950.0136, model+hostp950.193,44underruns.
Graphs really capture main+depth: framep950.0300s, decoderp950.00359,
model+hostp950.0264, zero underruns,3.68s audio in1.851s render wall.
FirstPCM0.510s is from replayed text availability, not a fresh backend/user request.
Warmup2.748s and service preparation6.697s remain separate. This supports moving
to a continuous-listening component probe, not full duplex qualification.
Streaming-ASR source/model metadata are pinned; all runtime assets already cached,
but checksum/inference checks remain. No ASR reference words may enter inference.

Plain words:streaming listening probe is prepared separately. It loads the pinned
1B ASR and identical32-codebook Mimi through complete strict audited loading, keeps
weights frozen, and feeds each1920-sample user frame only after its80ms arrival
boundary. No frame is discarded. After warmup/reset, three approved TRAIN inputs
008/019/020 are replayed at real arrival pace; all raw token/head outputs, delays,
backlog, samples and tensor hashes are retained. Measured reference user words
are parsed only after recognition for grading. Endpoint baseline follows publisher
head2/class0>0.5 plus two nonlexical frames; it is not qualified conversation policy.
No assistant audio, reference transcript, controller/backend or microphone enters
this component. A successful recognition run would still leave continuous
listening/speaking integration, cancellation and task-completion evaluation open.

Plain words:before submitting ASR, source inspection found that standard LMGen
replaces its offset0 input with initial tokens. Feeding real audio there would
omit the first80ms from the temporal model. The explicit ASR protocol advances
that single initial-only LM slot using initial IDs and zero actual PCM, then
checks all32 actual user rows exactly at every temporal forward. Mimi itself
receives no artificial prefix. Metadata records the virtual bootstrap separately.
Two real-Moshi CPU regressions must reject the unbootstrapped first frame and
accept every real frame across reset/bootstrap. These guards run before loading
the full ASR checkpoint. No parameter, audio sample or reference label changes.

Plain words:ASR969 PASSES both real-Moshi clock guards and complete strict model
loading, then all3 paced TRAIN inputs with worderror0. All32 provided user rows
match actual temporal input on every real/flush frame. Computep95~0.0122s;
maximum arrival backlog0.032s (other two<0.0001s). Raw head2/class0 plus two
nonlexical frames is NOT an accepted turn-end policy: it splits lightning into
3fragments and clarification/cancellation into7each. Do not dispatch tools from
that readout. A separate causal CPU policy replay now requires terminal decoded
punctuation and three quiet audio frames, retaining text across rejected interword
pulses. It also reports acoustic speech starts separately from semantic task
cancellation. Three local state/sequence regressions pass; replay on actual
captured model tokens, raw heads and PCM remains to run. No model decoding or
optimization is repeated; no reference words enter the policy.

## 2026-10-18 — concurrent hybrid development probe prepared

Plain words: endpoint replay 7511037 completed on the frozen actual streaming ASR traces. The conservative text-plus-quiet policy produced complete utterance counts 1/3/3 instead of raw head counts 3/7/7, with no leftover text and no reference words supplied. This is reused development policy replay, not fresh live acceptance.

The next bounded probe feeds actual TRAIN user PCM to the verified streaming recognizer while an external gpt-6-luna controller executes read-only record/cancellation tools and a dedicated pinned Kyutai service renders its words in single GPU steps. Both frozen services share one process/GPU. Arrived input frames take priority; actual backlog, packet availability and causal interrupted playout are retained. A declared 12-second minimum lookup delivery delay exercises result/cancellation races; it is not network latency. No optimizer or native model update occurs.

Local checks: 23 targeted lifecycle, endpoint, packet-clock and interruption tests pass. Completed controller dispatches can now close explicitly without pretending cancellation; closed or superseded requests cannot be delivered or cancelled. The renderer follows the pinned upstream drain-plus-flush completion rule. First run is the single reused TRAIN cancellation conversation, followed by broader development checks only after inspecting actual results. Native reasoning, microphone AEC, fresh holdouts and full-system qualification remain open. Interrupted ordinary answers currently resume from their beginning and may repeat a heard prefix.

## 2026-10-18 — first actual concurrent hybrid run 7511107

Plain words: source9ebcb65 completes the TRAIN020 cancellation replay on one GH200 in one process, with strict frozen ASR/TTS loads and no reference transcript supplied. Actual recognition word error0; every real input sample processed, ASR p950.01835s per80ms, maximum input backlog0.00110s. Actual task state marks the private lookup cancelled; completed responses are the cancellation confirmation and the resumed lightning/thunder explanation. The ordinary answer is interrupted by incoming speech, then restarted from its beginning after cancellation. Actual causal playout has zero underrun samples and exact accounting:430080 generated,425511 played,4569 cancelled before playback. Wall41.41s includes22.4s user recording and explicit tail. Controlled lookup delivery minimum12s is a test condition.

These are first reused TRAIN execution results, not full qualification. Next: verify actual cancellation receipt and absence of private PCM; independently transcribe only material retained in actual playout, with full-text word-error scores withheld for interrupted fragments. Then run all3 bounded development cases. No native model intelligence gain, microphone/AEC acceptance, owner voice approval or fresh holdout is established.

## 2026-10-18 — required semantic action protocol prepared

Plain words: first concurrent7511107 has an independently verified actual cancellation receipt for turn-1, exact reproduction of causal PCM including the4569 discarded samples, and no speech generation ever started for the cancelled private result. Source model intelligence is external. Independent played-speech audit7511132 and all3 TRAIN concurrent replay7511133 use sourcematched dc45446; retrieve terminal results before claiming speech quality or broader behavior.

A separate opt-in controller action protocol removes the optional JSON planning response by requiring one model-selected strict action, including a no-side-effect answer/clarify action. This preserves actual tool guards and keeps original behavior as the default. It does not force the model to look up or cancel. Twenty-five targeted local checks pass. See docs/research/2026-10-18-controller-action-contract.md for measured3-call cancellation overhead and fetched official API sources. Compare actual tool/latency results; no performance claim from unit checks.

## 2026-10-18 — broader hybrid replay failed closed before project identification

Plain words: three-case7511133 is FAILED, not passed. TRAIN008 completes its ordinary speech; TRAIN019 dispatches exactly “What's the locker label?” at2.734s. The actual controller selects a private read without user identification and the unchanged project guard throws WorkerContractError before any record execution. This is a semantic/controller failure. The first version retained the exact user/context trace but not rejected tool arguments; the next probe now saves the structured error's actual API output and arguments before failing. No guard is relaxed and no scripted clarification is counted as model behavior.

Independent actual-played-speech audit7511132 completes on prior7511107: resumed science answer WER0; cancellation text differs only canceled/cancelled (raw WER.25 retained); interrupted370ms fragment transcribed “light.” and gets no full-text WER. All bounds valid, exact interrupted playout reconstruction. No private-result speech started in that source. This does not qualify the broader failed conversation test.

## 2026-10-18 — required-action three-case execution and new evaluation inputs

Plain words:7511158 source12e9a42 completes all3 reused TRAIN conversations on one GH200. RecognitionWER0 throughout; all actual user rows reach the temporal forward. ASRp9512.67–13.26ms/80ms, maximum backlog4.56ms, zero causal playback underruns in every case.019 asks for identification, then speaks the actual counterfactual copper-kite result after tool execution while also answering evaporation.020 cancels and retains/resumes the unrelated science answer. Independent actual-played-speech audit submitted separately; do not equate controller text with heard speech or qualification.

Prepare7 new evaluation-only user recordings, not training examples or assistant teachers: unseen condensation/general knowledge, a delayed delivery-window record plus ordinary question, missing access-code identification, cancellation while retaining ordinary work, explicit negation, a backchannel and project correction. Only user clips are synthesized with the existing pinned TTS model; record worlds are declared separately. Input audio must independently pass per-utterance WhisperWER<=.05 and actual timestamp bounds before causal replay. Reference text stays separate and is first read for model scoring after inference. Known architectural gaps such as full-sentence repetition on resume and non-semantic acoustic interruption remain visible; this is not a claim that all7 should pass. Any case subsequently used to change the candidate becomes development evidence, requiring a further fresh check.

## 2026-10-18 — fresh replay adapter prepared without candidate behavior changes

Plain words: fresh user-input generation7511189 and independent quality7511190 use immutable8f82a30. Next replay adds an explicit evaluation-only mono-input profile with hash-bound independent approvals. The actual ASR, speaker, controller and endpoint modules are unchanged from the successful required-action development candidate, and their source hashes are retained in the new report. Fresh references are not supplied to recognition or controller; record values enter only through actual tools. Failed cases preserve error/API/input/packet evidence and are reported separately while the remaining evaluation cases run. A completed execution is never automatically task qualification.

The speech auditor can explicitly inspect completed cases from a partly failed suite, recording every excluded failure. Its default still rejects an incomplete suite. This opt-in diagnostic must never be represented as a full-suite pass.27 targeted local checks pass, including incomplete/nonfinite/changed-source quality rejection. No new training or shared-environment changes.

## 2026-10-18 — fresh input quality exclusions retained, five cases dispatched

Plain words:7511189 generation and7511190 independent input quality both finish. Five scenes000–004 pass every user utterance with WER0 and valid bounds.005 backchannel is rendered/recognized as “Uh-huh” rather than supplied “Mm-hm” (WER1), and006 correction is recognized as “metal result” instead of “Meadow result” (WER1/13). Both remain unapproved; the.05 threshold is unchanged. Frozen candidate source modules are byte-compared against12e9a42 before dispatching only the5 approved cases. Neither excluded clip is counted as a model failure or silently evaluated.

A separate re-recording option generates ONLY those2 scenes with coral user voice and an explicit new “Uh-huh” backchannel input. Correction wording remains unchanged. These are new source audio and references with separate hashes and fresh independent judging, never a relabelling of the rejected audio. Original batch, quality failures and timestamps are retained. The existing5-case GPU run uses its original immutable code/quality proof; no source or fixture changes under it.

## 2026-10-18 — first five fresh conversations execute; repeated-answer defect remains

Plain words:7511217 completes all5 independently approved new scenes000–004, with recognitionWER0, no lost actual user samples and zero causal underruns in every case. Controller texts correctly explain condensation and moving shadows, answer rectangle corners, use real Tuesday-afternoon/falcon-nine receipts, clarify missing identity and honor explicit no-lookup negation. Independent actual-played-speech audit is still required. Case003 exposes a conversation-control defect: the cancellation reply also answers the unrelated question, then the retained ordinary answer repeats it. Do not call this natural conversation or full-system qualification.

Re-recording7511242/quality7511244 approves the new Uh-huh backchannel005 with WER0 but still excludes006: “Copper West” is recognized as “Copperwest,” WER2/13. Original references are unchanged. A distinct correction scene007 uses Oak Central/Pine Central and the same meeting-room correction intent; it needs its own generated audio, references and independent quality check. This is a new source scene, not relabelled failed audio. The frozen candidate's controller/endpoint/ASR/speaker modules remain unchanged during these evaluations.

## 2026-10-18 — explicit owed-speech state candidate prepared

Plain words: the observed repeated-answer failure in fresh003 becomes development evidence for a separate state-aware controller candidate. The client snapshots actual ordinary replies that are queued or interrupted, with their request IDs and actual generated text. They are not lookup tasks, tool receipts or known heard-word positions. The controller explicitly lists which queued ordinary answer its new reply fulfills (or which the user withdrew); the client removes only those validated IDs. A cancellation-only reply leaves unrelated answers queued. There is no string deduplication or scripted replacement answer.

The original required-action candidate remains unchanged in immutable worktrees while its remaining fresh tests run. New state-aware behavior is opt-in.31 targeted checks pass: real cancellation receipt remains necessary, ordinary work is not cancelled by speech-state consumption, unknown/private/duplicate speech IDs cannot be removed, and conflicting snapshots reject. Actual replay and independent heard-speech audit are required before claiming the repetition problem fixed. Cases used to design this change are no longer fresh qualification for it.

## 2026-10-18 — historical input approval retained for candidate comparisons

Plain words: changing a candidate must not re-record or relabel its input. A separate loader can execute the exact original quality validator from its clean immutable worktree in this same Git repository. It checks shared repository identity, exact commit, cleanliness and file bytes before invoking the original unchanged source/quality/hash checks. The report records that validator's commit and SHA. This permits the new speech-state candidate to replay original fresh003 as explicitly DEVELOPMENT evidence, without loosening the input gate or changing the frozen first evaluation. Unknown commits and incomplete source specifications reject before reading input artifacts.32 targeted checks pass. The actual GPU replay must still exercise the compatibility path.

## 2026-10-18 — actual correction counterexample and atomic replacement candidate

Plain words: frozen required-action7511514 on independently approved new007 has recognitionWER0 and executes the cancellation, but says it cannot check Pine Central. It never reads Pine; no replacement result is delivered. This is a failed correction task despite successful process completion. The one-tool contract allowed cancel or read, but lacked an action that did both for a corrected request.

A separate state-aware candidate now exposes replace_lookup: validate current revision, new identified project/field and every selected old target before any mutation; under one lock, cancel only those old tasks and perform the actual new read. Its compound receipt includes both real cancellation and record read. Delivery/readiness guards flatten that actual nested receipt, so a later cancelled replacement is still suppressed. Unknown fields, stale revisions, malformed old target lists, own-dispatch targets and missing user identification cannot cause a partial replacement.37 targeted checks pass, including actual controller protocol and nested late-cancel tests. Actual007 replay and original conversation regressions are still required. This is application/controller behavior, not native learned delegation.

Earlier state-aware1506 completes3 TRAIN scenes with recognitionWER0/zero underruns. Development0031512 completes with exactly one combined cancellation/ordinary response and no queued repetition; independent played-speech audits submitted separately. No backchannel/ASR omission fix is claimed.

## 2026-10-18 — acoustic detector compatibility screen, no adoption

Plain words: MaAI's English current-backchannel detector is a relevant pretrained component because fresh005's acknowledgment never reaches the transcript. Exact source/HF/LFS pins and primary documentation are recorded in docs/research/2026-10-18-backchannel-detector-screen.md. Prepare a CPU tensor-only asset inventory; do not import its desktop stack, change shared Transformers, use upstream strict=False as proof, classify audio or claim model/license approval from the inventory. Model rights are unspecified in its public weight repository and separate from MIT code. Current hybrid control/correction jobs continue independently.

## 2026-10-18 — explicit lexical-interruption operating-point probe prepared

Plain words: retain acoustic interruption as the default. A separate opt-in lexical policy keeps an active answer running until a real, recently voiced ASR lexical token arrives (or a complete request is dispatched). It does not classify backchannels and cannot recover acknowledgments omitted by ASR; the known omitted Uh-huh should simply stop causing an acoustic-only restart. Because recognition is delayed, genuine interruptions will stop later. Measure this, do not claim GPT-Live-like interruption latency from fast GPU compute. New queued speech also waits through a declared660ms recent-voice/ASR-tail barrier; current ongoing speech may overlap the user during that wait. No future input or expected transcript enters the policy.37 existing targeted checks pass; actual concurrent backchannel and genuine cancellation/regression replays are required for this change.

MaAI remains only a pinned CPU asset inventory: its public weights lack model-license metadata, its current inference stack differs, and no pretrained detector is adopted. The lexical policy is a measured fallback operating point independent of that investigation.

## 2026-10-18 — access-code speech surface control prepared

Plain words: fresh002's actual controller response contains curly double quotes around falcon nine; independent playback ASR reports “UMFALCON9” and rawWER.333. Do not call that an intelligible-code pass from controller text alone. The pinned upstream TTS preparation normalizes curly apostrophes, colons and parentheses, but leaves curly double quotes in per-word tokenizer inputs. This motivates a matched diagnostic, not a claim of cause.

The existing strict streaming renderer now optionally replays a fully played actual hybrid record response, verifying its source WAV hash, actual read receipt and completed response text. Compare verbatim versus removing only balanced curly double quotes, with identical assets/voice/graph mode/warmup and paired random seeds. Preserve original backend text, actual rendered text, source hash and per-word tokens. Letters/digits/apostrophes, straight inch marks and unmatched quote characters remain unchanged. Default live speaker/coordinator behavior is unchanged. The diagnostic output remains explicitly replayed, unqualified and not training approved; independent speech audit is required before adoption.

## 2026-10-18 — detector tensor inventory complete; strict head-only proof prepared

Plain words: CPU7511739 completes a tensor-only inventory of the pinned17MB MaAI detector: all71 tensors finite, no unsafe pickle globals. Its checkpoint contains a shared512x512x3 rate-conversion layer,512→256 projection and the two-channel transformer/classifier; frozen Mimi encoder weights are separate. This proves asset contents only.

Prepare an isolated CPU head proof importing only exact Git-pinned neural modules, excluding the desktop package entry point. Register every checkpoint tensor and load strict=True with tensor equality. Check cached versus full random-feature inference, future-perturbation prefix invariance and sensitivity to each current input channel. Synthetic features prove no audio behavior, encoder parity, English threshold or deployment rights. No shared package changes, threshold adoption or controller wiring. The head's registered conversion parameters are unused at native12.5Hz, matching the upstream encoder path; frozen Mimi still requires separate verification if pursued.

## 2026-10-18 — separate self-hosted text intelligence component screen

Plain words: external gpt-6-luna remains the current hybrid brain; native7B speech weights have not gained sufficient ordinary intelligence. A text-only4B Qwen instruct component may provide local knowledge/tool selection without old30B layer/tick changes. Exact public Apache2.0 assets are pinned. CPU imports on current SIF already succeed with existing Transformers4.55.3; prepare hash/index/template checks without changing packages. No model inference or training is established by the asset check. If pursued, typed controller tests must precede voice integration and cannot count as native speech-model intelligence or full-system acceptance.

## 2026-10-18 — detector head CPU proof complete; exploratory shadow variant prepared

Plain words:7511928 strict-loads all71 detector tensors exactly, without the desktop stack. Cached/full random-feature probability error1.19e-7; future-perturbation prefix error0; either-channel perturbation changes outputs. This establishes head loading/causality only.

A separate unpaced shadow probe scores the completed actual lexical-policy recordings1790/1791. It uses the already independently verified official Moshi Mimi continuous encoder at raw24kHz, explicitly DIFFERENT from the publisher's HF/16kHz-resampling path; feature parity is not asserted. The frozen head receives only current/past acoustic features of user and actually played assistant PCM. Source annotations are read only after all scores. Preserve feature tensors, per-frame scores, source hashes and throughput; no threshold, speech control, deployment/model-license approval, training or full-system qualification follows from this diagnostic. Shared packages and the working speech runtime remain unchanged.

## 2026-10-18 — typed local controller proof prepared, no voice substitution yet

Plain words: after the pinned4B assets/import check, a bounded GPU proof will load the complete text model with strict=True, explicit true tied-embedding aliases only, and equality against every checkpoint tensor before device copy. No optimizer or remote API. Eight typed component cases cover ordinary knowledge, missing identification, actual private read, cancellation, correction replacement and explicit no-lookup negation. Private values stay out of the first prompt and enter only through actual read-only execution receipts. Raw model calls, rejected formats, first-token/full-token availability and real task states are retained. No action-format repair or canned answer fallback.39 targeted local guards pass. These are typed NLP component tests, not streamed speech, native duplex intelligence or a fair audio-model comparison; semantic review still follows actual outputs.

## 2026-10-18 — local4B typed baseline: knowledge works, two contract failures

Plain words: CPUassets7511995 passes published hashes for8GB weights and actual imports. GPU7512030 source304d178 completes its typed8-case probe in64s. All398 source tensors match exactly; only the true tied lm_head alias adds the399th model-state entry;4,022,468,096 parameters frozen, no missing/unexpected keys. Triangle correctly says three, pencil correctly graphite, lightning explanation and no-lookup negation are correct. Actual identified read and cancellation execute. Missing-project asks the tool anyway and is blocked before execution; correction cancels only and omits the new read. Action contract6/8, NOT qualified. First token~48–69ms; full model reply1.0–2.14s, two-call tool replies longer. Typed text performance does not establish voice latency or native speech intelligence.

Next optional prompt profile illustrates missing identity even when only one record name exists, replacement versus cancel-only, using dummy names and no result values. Illustrations are explicitly not conversation history or executed receipts. Actual client state and original user still control the unchanged identity/action guards. Weights and all8 typed cases remain fixed; no training. Compare actual outputs/receipts before integration, then new held-out tasks. Defaults and baseline artifacts remain unchanged.

## 2026-10-18 — local controller examples pass reused typed cases; worker proof next

Plain words:7512050 uses the same frozen4B model and same8 development cases as2030, adding explicitly illustrative contract examples. All8 action checks pass, including missing identity and atomic replacement. Ordinary triangle/graphite/lightning replies are correct; actual private results come from executed receipts. This is prompt improvement on reused typed cases, not fresh evaluation, training, or voice qualification.

Prepare a strict local executor behind the unchanged worker guards. Its internal normalized envelopes are explicitly local, with raw generations/timing/provenance retained; no API or fallback. Final reply formatting uses a side-effect-free function and validates all required fields, including consumed pending-speech IDs. Run the actual state-aware worker on the reused8 cases before2GPU audio integration. Speech services remain together onGPU0; local controller is disclosed onGPU1. Independent audit must accept this explicit topology without falsely claiming all components share oneGPU.21 targeted worker/schema tests pass; compile and shell syntax pass.

## 2026-10-18 — actual local worker proof rejects premature integration

Plain words:7512153 source278dc55 completes. Ordinary triangle/graphite/lightning and negation pass, but missing identity attempts a read and is blocked; actual private/cancel receipts execute but final model output is plain prose and strict output validation rejects it. Replacement fixture also omitted reserving the current task, so that result is invalid rather than evidence against model replacement. No audio integration submitted. Preserve raw outputs and executed receipts.

Correct the proof fixture reservation. Make the local transport match the successful typed probe more closely: reply function first, illustrative contract examples before the actual state, and an explicit post-receipt user instruction requiring the reply function and every required field. No generated output repair, fixed answers, relaxed identity guards, or schema changes.31 targeted worker/format/replacement/speech-state/validator tests pass. Re-run actual worker proof before audio.

## 2026-10-18 — local worker7/8; structured local prompt state

Plain words:7512181 sourceedc2622 fixes final reply function formatting and properly reserved replacement: actual read, cancellation and replacement all complete with required structured fields. Missing project still attempts a read and the unchanged guard blocks it. Seven of eight reused typed cases pass; no audio integration yet.

The local transport now receives an explicit internal state object (available project/field names, active task states, owed speech, protocol), rather than relying on an OpenAI-specific instructions string. Its concise local prompt preserves identity, actual-receipt, correction, duplicate-cancel and owed-speech rules with the same examples and actual state last, matching the successful simpler typed screen more closely. No private values or reference answers are injected. Default remote transport unchanged. Guard still blocks unidentified calls and raw model failures remain visible; this is another development prompt iteration, not model training or fresh evidence.

## 2026-10-18 — acoustic shadow complete; latency bottleneck and proof-only optimization

Plain words:7512028 completes all4 causal captures with all real samples and strict original head/codec weights. The omitted-ASR acknowledgment has peak detector probability.303, versus other scene maxima.059/.024/.007. Tiny development evidence only: no English threshold, publisher encoder parity, model-rights approval or speech-policy adoption. Isolated p95 grows12→29→675→1492ms across scenes, so current implementation is unusable as a real-time component.

Pinned source MultiHeadAttentionAlibi.mask_scores rebuilds a full square mask and assigns context exclusions in nested scalar loops at each longer cache. Prepare research-only equivalent vectorized query-row masking; preserve exact finite lower-triangle+1, ALiBi bias, causal/context boundaries and caches. CPU proof must compare exact score masks including lengths beyond250 and full cached model outputs before any audio timing comparison. No publisher source modification, checkpoint rewrite, shared package change or live speech wiring.

Plain words:local worker7512200 sourced881b2f now passes8/8 reused actual contract cases. First paced TRAIN020 local ASR/TTS GPU0 + Qwen GPU1 job7512210 and independent actual-played audit7512212 submitted from byte-verified immutable source. No remote API/fallback; no full qualification implied.

## 2026-10-18 — next user-input evaluation batch prepared

Plain words:prepare8 new synthetic USER-only scenes008–015: arithmetic, private lookup plus ordinary science, missing identity with two available projects, cancellation preserving an explanation, corrected project, negated lookup plus logical converse, acknowledgment and genuine explicit stop/shorten. Generate separate references; independent user-clip ASR quality must pass the unchanged WER<=.05 and bounds checks before inference. No assistant teacher or training approval. This batch is not yet evaluation evidence. Once results inform a change, label subsequent reuse development. Local fresh wrapper explicitly uses2GPUs and no remote controller credentials; source model/assets and input-approval provenance stay mandatory. Baseline candidates and old quality-validator worktrees remain intact.

## 2026-10-18 — local audio execution succeeds, ordinary-task completion fails

Plain words:first actual local audio2210 completes TRAIN020 with recognitionWER0 and actual cancellation. Independent2212 reconstructs causal playout, hears partial 'light' plus full cancellation (rawWER.111 spelling). The controller wrongly consumes owed turn-2 while replying only that the lookup was cancelled: science answer is dropped. This is a real task-completion failure, not ASR or ordinary knowledge failure. The earlier8/8 action checks did not qualify this speech-state behavior. No full-system pass; original files retained.

Add explicit dummy speech-state illustrations to the LOCAL prompt only: cancellation of a lookup alone leaves ordinary speech IDs unconsumed; answering that explanation in the current reply or explicit user withdrawal permits consuming it. No fixed factual reply, generated output repair, guard relaxation or weight update. Strengthen the typed cancellation fixture to require preserving owed ordinary speech, then repeat actual audio only if that check passes. Evaluation remains development; new008–015 user inputs are still being independently quality checked.

## 2026-10-18 — detector mask CPU exactness proven; bounded audio comparison prepared

Plain words:CPU7512215 source2d9feb6 passes10 exact attention-mask comparisons (contexts8/250, lengths1/17/261, single/multiple/full query rows). Full37-frame cached head probability difference0 with the vectorized mask. Strict head loading unchanged. This proves numerical equivalence on declared probes, not audio behavior.

Optional shadow comparison requires that exact proof and original2028 artifacts. Re-encode same arrived user/actually played assistant audio; assert source hashes and every archived continuous feature exactly equal, and probability differences<=1e-5. Preserve original versus optimized p95. No threshold selection, speech-control integration, publisher feature-parity or deployment claim. Default shadow implementation remains original. The unresolved weight-rights boundary still applies.

## 2026-10-18 — actual multi-turn local format failure; generation-prefix variant prepared

Plain words:strengthened typed7512224 passes8 reused cases including cancellation leaving unrelated owed speech intact. Broader TRAIN audio7512232 nevertheless fails on a rejected local function serialization. Actual multi-turn behavior remains unqualified; dependent speech audit was cancelled. New6-case2234 is a separate measurement, not acceptance. Preserve full raw failure before interpretation.

Prepare an explicit generation prefix '<tool_call>\\n' for LOCAL action requests. This is supplied to the model BEFORE sampling; it selects no function, arguments, factual answer or pending IDs. Raw generated continuation and supplied prefix are recorded separately; parsing validates their concatenation strictly. No post-generation rewriting or prose-to-action repair. Default standalone typed baseline has empty prefix. Repeat actual worker and multi-turn audio tests; any reused new scenes become development evidence for the changed candidate.

## 2026-10-18 — exact acoustic optimization complete; isolated text attention screen prepared

Plain words:GPU7512227 re-encodes all4 identical causal recordings. Every continuous feature matches original2028 exactly and every detector probability difference is0. Vectorized p95 is13.36/15.24/17.07/17.65ms versus original12.04/28.60/675.13/1492.00ms. This removes the diagnosed implementation bottleneck; it does not qualify English detection, select a threshold, establish publisher encoder parity, authorize model rights or change the speech policy.

Prepare an isolated opt-in Qwen typed-worker comparison of explicit PyTorch flash attention versus the current math backend. Strict frozen weights, generation prefix and worker guards unchanged; no fallback if flash is unsupported. Record selected backend and actual tokens/timing. Default math and current audio candidates remain unchanged. This is latency work, not extra model training or an assumed capability gain.

## 2026-10-18 — new six-case full-task failures separate intelligence from control

Plain words:7512234 sourcee645198 completes5/6 new conversations and captures011 format failure. Arithmetic008 correctly reaches10; logical-converse013 correctly rejects all-birds-are-robins; actual project correction012 speaks only Thursday afternoon.009 incorrectly repeats the delivery lookup for the latest ice question, and015 gives the requested short answer then resumes the superseded long version. These are real full-task failures despite recognitionWER0 and mostlyWER0 speech audit2235. Strict function serializer additionally rejects011 multiple/extra output. No overall qualification. These6 scenes are now development for any changed candidate.

Local concise prompt had omitted the original worker's latest-user-only emphasis. Restore it explicitly, reassert the actual latest ASR user text, and omit redundant nested task-context copies from prompt state (actual IDs/revisions/operations remain). Explicitly describe a requested shorter replacement as superseding owed long speech. Strengthen typed worker proof with actual multi-turn ordinary-while-private-pending and superseded-long-answer cases; original standalone8-fixture screen unchanged. Recorded pre-generation prefix already passes original8 checks in2301, but does not establish multi-turn success. No weights trained, semantic guard weakening or answer repair.

## 2026-10-18 — additional untouched user-input scenes prepared

Plain words:prepare new evaluation-only016–023: missing storage-label identity with two projects, brief Right acknowledgment, stop/shorten to first step, private lookup plus heat-conduction question, cancel one of two lookups, explicitly replace an ordinary explanation with arithmetic, conversational memory, and unavailable private project. New clips and independent input-quality receipts are mandatory. No assistant targets or training. Do not claim the acknowledgment/genuine-interruption scenes exercised overlap until actual playback timing confirms it. Original010 rejects color/colour rawWER.2;014 actually repeats Okay three times and rejectsWER2. Keep both failures; new scenes are distinct, not relabelled approvals.

## 2026-10-18 — older4B control remains unqualified; isolated newer9B compatibility prepared

Plain words:2314 math and2316 flash both pass8/10 strengthened worker cases. Missing identification regresses to attempted read; shortened explanation still leaves owed old ID unconsumed. Latest-user ordinary-while-pending now answers evaporation correctly. Flash reduces many stage times by~15–20% but is not a control improvement. No new local audio submitted from this candidate. Separate untouched user batch016–023 generated2342 /quality2343 pending.

Prepare an isolated CPU compatibility screen for Qwen3.5-9B after checking its official card/config. Exact model and ARM64 wheel pins committed; no shared dependency changes. This directly tests a stronger pretrained controller foundation instead of assuming additional fine-tuning compute or more wording changes unlock reliability. It remains separate from the rejected native speech weights, requires actual strict-load/task/latency evidence and does not count as native speech intelligence.

## 2026-10-18 — newer runtime first CPU check rejects inherited regex

Plain words:CPU2351 installs exact four wheels into a separate run directory, then fails before model inference because Transformers5.5.3 requires regex>=2025.10.22 while inherited regex is2025.9.18. Shared speech packages remain untouched. Pin a fifth ARM64/Python3.12 wheel, regex2025.11.3, into the same isolated package inventory; repeat actual import. No version-check bypass or metadata purge. Third user batch2343 approves7/8 scenes016–021/023;022 rejects favourite/favorite spelling plus repeated question. Keep it excluded.

Plain words:exact regex wheel resolved and added after the first filename-selection attempt failed before editing pins. The prior commit records the planned fix; this commit contains its actual pin/runtime changes. Prepare the separate model asset stage, requiring verified runtime bytes before imports/download: all14 public objects, Git/LFS hashes, complete tensor-index inventory, actual config/model-class/tokenizer imports and non-thinking template check. No model inference or package changes in that asset stage. Disable optional Xet transport to avoid introducing its binary dependency into this isolated check.

## 2026-10-18 — newer model class imports; initialization context moved upstream

Plain words:CPU2554 gets past all package checks and successfully imports Qwen3_5ForConditionalGeneration. It then rejects the old4.x location of no_init_weights. Exact installed5.5.3 source places that context in transformers.initialization; update both isolated compatibility and asset stages to the real API. No loader checks disabled or shared dependencies changed. Dependent assets2555 never ran, so no GPU/model inference claim.

## 2026-10-18 — newer controller strict-load and native-format worker proof prepared

Plain words:prepare a typed10-case actual worker proof for the newer9B model, gated on the separate runtime and full asset receipts. Load every whole-model tensor strict=True, compare every source tensor onCPU before device copy, preserve source dtypes and freeze all weights. No vision inference or selective missing-module skip. Use the published tokenizer template with enable_thinking=False and its native XML-style function format. Parse exactly one named function and schema-validated parameters; reject prose, multiple calls, unknown/duplicate/missing arguments and malformed arrays. Record pre-generation prefix separately, raw continuation/timing, provider/model and actual receipts.33 targeted local contract/parser tests pass. This prepares a component test, not capability or speech qualification; no new audio integration yet.

## 2026-10-18 — newer runtime/assets pass; first GPU wrapper catches missing import

Plain words:CPU2611 passes actual model class and initialization imports using five isolated pinned wheels. Asset2612 completes14 public objects/775 tensor headers with exact Git/LFS hashes, complete index and non-thinking tokenizer check in4m07. No inference yet. GPU2652 stops before model construction because the new wrapper missed its time import; correct that source error. No capability claim.

Exact verified upstream5.5.3 Qwen3_5PreTrainedModel explicitly ignores '^mtp.*' on normal loading. The pinned checkpoint includes15 auxiliary speculative-decoding tensors beyond the normal conditional-generation architecture. Account for this exact fixed inventory instead of dropping it: retain these source tensors in frozen storage modules under their original names, then strict=True load and compare ALL775 source tensors. Any other unexpected tensor fails. This does not implement/run MTP or vision inference; both limitations appear in the proof. Core inference weights are unchanged, no random head inserted, no strictFalse or skipped source tensor. Prepare the next actual GPU load proof.

## 2026-10-18 — stronger baseline new evaluations and explicit read capabilities

Plain words:frozen external-controller source740c9ac job2353 completes6 new008/009/011/012/013/015 conversations with recognitionWER0. Actual text/tool behavior addresses arithmetic, simultaneous science/private work, cancellation preserving mirror explanation, corrected project, logical converse and shorter replacement. Independent2354 retains speech uncertainty: possessive segmentationWER.111 on009 and 'bird smeans' WER.1 on013; interrupted fragments are not full-text scored. No overall speech/microphone qualification.

Second new batch2569 completes6/7 approved016–021/023, recognitionWER0 in completed cases;023 attempts an unavailable project and guard rejects.017 'Right' interrupts/restarts the seed explanation, a natural-backchannel FAIL despite correct resulting words.019 private silver-square speech is uncertain ('Shilver Square', WER.111).020 and021 text behavior appears right but inspect actual cancel receipts/consumption before scoring.2570 audits actual completed playout only. Keep failures and partial-suite disclosure.

Prepare an opt-in read-tool capability scope: expose lookup/replacement only for available projects actually named in USER history; constrain exposed project/field enums and retain all execution guards. When no authorized available read exists, model still chooses its own clarification/unavailable reply. This avoids offering impossible reads; it is an explicit application constraint, not evidence that the model independently made the right authorization decision. Defaults and preserved baseline740c9ac unchanged.34 targeted tests pass; no new evaluation result yet. These reused cases are development for the changed scope.

## 2026-10-18 — all775 newer-model tensors exact; two actual transport bugs found

Plain words:GPU2709 source71f4781 strict-loads and compares ALL775 source tensors exactly,760 normal conditional-generation tensors plus15 explicitly retained unused MTP tensors;9,653,104,368 parameters frozen, no missing/unexpected keys. Text-only use, no vision/MTP speculative inference. Runtime uses pureTorch fallback for missing optional fast kernels; no automatic package installation.

The ten-case probe gets5 nominal passes but exposes integration bugs rather than five established reasoning failures. The model initially asks the correct missing-project clarification and gives the correct evaporation response, then continues inventing further user turns AFTER emitting its actual end-of-turn token. Official config stores eos_token_id248044 in nested text_config; whole conditional-generation config has no top-level EOS, so from_config generation did not stop. Explicitly use the pinned tokenizer EOS in Generator.generate. Do not truncate or repair already generated output. Private/cancel/replacement replies fail during chat-template rendering because the published3.5 template expects function.arguments as a mapping; the older3 template accepted serialized JSON. Add an explicit history-format hook, retaining identical model-selected values and actual normalized output strings. No semantic overrides, invented receipts or relaxed parsing.34 local tests pass; repeat actual ten-case proof before interpretation or audio.

## 2026-10-18 — newer9B actual worker9/10; explicit speech actions comparison prepared

Plain words:GPU2934 sourcebfe568a fixes EOS/history transport and passes9/10 actual typed worker cases. Correct missing identity, actual read/cancel/replacement, latest ordinary question and shorter supersession. The remaining cancellation case wrongly consumes unrelated owed ordinary speech while merely confirming a private cancellation. No local audio integration approved. Typical model stages1.7–3.8s before speech; no low-latency claim.

Compare an explicit local output-action interface with the original consumed-ID array. respond_to_user means reply while preserving all owed ordinary speech. A separate resolve_pending_speech means reply AND finish/withdraw/supersede explicitly selected ordinary speech IDs, constrained to actual pending IDs. The model chooses the action and IDs; the adapter maps its published action schema to the unchanged canonical worker schema and records both native/canonical actions. No answer rewriting, semantic decision override, fabricated receipt or skipped validation. Defaults preserve the original array protocol.35 targeted tests cover preservation, exact selected-ID mapping, invalid/empty IDs and unchanged original schemas. Compare actual ten-case behavior before audio.

## 2026-10-18 — explicit speech actions10/10; isolated resident transport prepared

Plain words:GPU3070 original array stays9/10;3071 explicit actions passes ALL10 reused typed actual worker cases, including cancellation preserving unrelated ordinary speech and shorter replacement resolving the old answer. Same frozen9B weights and unscoped private-read guard; this is not fresh or audio qualification. Scoped external unavailable-record3098 now produces its own honest unavailable reply, no read receipt;3099 actual-played ASRWER0. This fixes one application capability boundary, not model-only authorization intelligence.

Prepare a resident local controller subprocess so the speech process retains verified Transformers4.55.3 while only the child uses pinned5.5.3. Private mode0600 Unix socket, bounded JSON messages with request IDs, full child logs, source/assets/device/tensor-proof handshake, exact model/provider and preserved errors. No HTTP/API/fallback. Child actual model chooses normalized actions and IDs; parent executes guarded tools. Parent version checked after every response and child cleanup bounded. Typed resident proof must precede2GPU audio integration.38 targeted tests include packet integrity, identity mismatch, failure provenance, unchanged responses and parent-version protection; mocks do not establish actual model inference.

## 2026-10-18 — actual resident controller10/10 with separate runtime; audio integration prepared

Plain words:SSH timed out after the resident submission; durable receipt confirms job3324, so no duplicate submitted. Actual GPU3324 completes all10 typed cases. Parent stays Transformers4.55.3; child5.5.3, exact775 tensor load, private socket/source/device handshake verified, no remote API. Startup27.4s; model is resident for subsequent calls. Child log retained. This proves actual isolated typed inference, not streaming audio or fresh task qualification.

Prepare first paced TRAIN020 integration with ASR/TTS together onGPU0 and resident9B controller subprocess onGPU1, explicit speech actions, same guards and actual PCM ledger. Parent keeps old speech dependencies. Source must pass Git/MD5 before submission and independent actual-played audit must check the disclosed topology. Child closes even if a replay fails. New wrappers are opt-in; existing external and4B defaults remain available.38 targeted tests/compile/shell syntax pass. Do not adopt solely because a process exits successfully.

## 2026-10-18 — first newer9B audio repeats pending read and loses mixed ordinary answer

Plain words:resident audio3381 completes TRAIN020 with recognitionWER0, parentASRp9513.79ms/maxbacklog89.9ms. It fails task completion: turn-2 reissues the Birch badge lookup and mixes its private value with the requested lightning explanation. Turn-3 correctly cancels both duplicate lookups, so the mixed science reply is discarded with the cancelled private response. Only cancellation completes.3382 hears cancellation (rawWER.091 spelling), no science. Typed10/10 did not predict this actual multi-turn failure; no adoption or native intelligence claim.

Hypothesis: the foreground transcript includes earlier user requests and spoken answers but lacks actual assistant tool-execution events, making an unspoken pending request look unhandled despite task metadata. Prepare a typed ablation of the exact captured actual ASR/context, same frozen9B model: omit versus include the actual prior call plus a client execution-status notice. The notice is projected from a validated actual receipt, hash-bound and deliberately withholds its value while the client holds delivery. It is not assistant speech, reference text, a fabricated tool answer or permission to expose delayed private data. Model-template mapping preserves real assistant/tool roles and IDs.40 targeted tests cover value withholding, mismatched receipt rejection, role projection and orphan-notice rejection. Test the hypothesis before wiring it into the paced loop; no training or existing candidate mutation.

## 2026-10-18 — isolated optional Gated DeltaNet kernel assets prepared

Plain words:the9B runtime explicitly reports a pureTorch inference fallback; model stages are several seconds. Prepare separate optional FLA0.5.2 and fla-core0.5.2 universal wheels, exact PyPI hashes, outside the verified runtime/speech package folders. CPU stage verifies every combined module byte, full installed inventory and inherited Torch/Triton/einops metadata. It does not import CUDA FLA operators or prove inference. No causal-conv1d build, driver change or shared dependency install. A later GPU variant must prove actual selected functions, strict identical weights, behavior and timing before adoption. References: https://github.com/fla-org/flash-linear-attention and the exact installed Transformers5.5.3 Qwen3.5 implementation. This is implementation performance work, not extra training or assumed capability gains.

## 2026-10-18 — actual tool-history ablation prevents the duplicated pending read

Plain words:GPU3642 sourcee05869d reproduces the exact failed3381 request from captured ASR and actual first-read state. Without tool history, the unchanged9B reissues the badge read and mixes private/science text. With the real prior function call and hash-bound value-hidden client execution notice, it chooses an ordinary reply, answers lightning/thunder, and performs NO duplicate read. Same weights, state, user text and execution guards. This is a single reused typed causal diagnostic, not fresh audio qualification.

Prepare opt-in live execution notices. The worker observes only an actual guarded operation after it executes, before its final reply. A nonblocking queue delivers these notices to the media coordinator; it drains them before new request snapshots, adds real assistant/tool roles, withholds private values and logs full source evidence separately. Direct ordinary replies and rejected operations generate no execution notice. Default context unchanged.42 targeted tests pass, including observer ordering and no notices on an unidentified rejected read. Retest paced audio before adopting the context change.

Plain words:pair each execution notice atomically with its world operation under the existing reentrant lock, so concurrent read/cancel notices cannot be reordered by a producer scheduling gap. The callback only enqueues metadata; the lock is released before any model request. Added concurrency check confirms another thread can reserve work during model reply formatting.42 targeted tests pass; no claimed live result yet.

## 2026-10-18 — optional kernel assets verified; matched GPU speed/behavior comparison prepared

Plain words:CPU3645 passes exact463 FLA module files and full package inventory for FLA/fla-core0.5.2. Inherited Torch2.9.0a0/CUDA13, Triton3.4.0, einops0.8.1, isolated Transformers5.5.3. This is provenance only; no CUDA operator import/inference in the CPU stage.

Optional GPU variant inserts this exact separate package before the model import and requires ALL24 Gated DeltaNet layers to actually select chunk/recurrent/gated-norm functions from that verified FLA path. Strict775-tensor source equality remains mandatory; convolution function selection and GPU hardware are recorded rather than assumed. Missing/incompatible kernels fail instead of silently substituting. First comparison is in-process typed10 cases, original versus optional kernels, same model/protocol/source. Default resident speech path unchanged. No inference-speed or numerical-equivalence claim yet.

## 2026-10-18 — fused-kernel first execution stops on container driver discovery

Plain words:3740 reaches first FLA CUDA warmup, then Triton3.4 searches an invalid image ldconfig entry and cannot find libcuda. No optimized behavior or speed result. Separate one-task interactive inspection3813 shows Torch already uses visible /.singularity.d/libs/libcuda.so.1 and both link names exist there. The first interactive inspection3763 accidentally inherited multiple Slurm tasks and timed out its container launches; always specify --ntasks=1 explicitly for srun. No application/model result came from that failed diagnostic.

Prepare an inference-process-only TRITON_LIBCUDA_PATH override to the ACTUAL mapped Torch driver directory. Require exactly one mapped library and matching hashes for its bare/versioned link names, then verify Triton's own search API selects that directory. No driver/library/cache/symlink edits, copied driver or shared environment change.44 targeted tests include mismatched/ambiguous driver rejection. Repeat only the failed optional-kernel variant; the existing matched Torch baseline remains valid because model/prompt/default execution are unchanged.

Plain words:history-enabled resident audio3703 now performs only one private read, cancels it and completes the separate lightning explanation.3704 independent scienceWER0; cancellation rawWER.154 includes Birch/Burt pronunciation and canceled/cancelled spelling, so voice remains unapproved. This supports the actual-history fix on one reused audio scene. Broader same-frozen-source6ea3d00 TRAIN3 and reused6+7 development batches, with independent speech audits, are submitted; retrieve receipts /private/tmp/duplex-qwen35-history-broad-submit-full.txt. No full qualification or new training.

Plain words: 18 October overnight — frozen history candidate 6ea3d00 completed
three TRAIN regressions7513806 and six reused second-batch scenes7513808. All
user ASR was exact. Third batch7513810 completed six of seven: unknown record
023 still rejected by the unscoped execution guard. Backchannel017 still
interrupts/restarts; this is a failure. Written arithmetic in008 became garbled
spoken symbols (audit7513809 WER0.266667), and020 narrated withheld-value
metadata. These motivate an opt-in spoken-output instruction, without changing
actual function arguments or repairing generated text afterward.

Plain words: optional frozen FLA kernels7514182 pass the same ten typed actual
worker cases, with median model stage2.100s versus matched torch baseline
7513739 2.688s (about22% lower). These are development typed cases, not full
conversation latency. Both strictly load all775 source tensors. Triton's driver
search is process-locally pointed at the hash-verified library already loaded
by Torch; no system driver or shared environment changed. Resident RPC now
accepts opt-in verified kernel assets and the spoken-output instruction, checks
both in its source/asset handshake, and leaves defaults unchanged. Full audio
acceptance remains open; no new optimizer steps or adopted native weights.

Plain words: resident7514733 passes10/10 typed actual worker contracts with
spoken-output instruction and verified FLA kernels; startup33.1s, parent speech
Transformers4.55.3 and child5.5.3 checked. Actual audio three-case7514741,
second-batch7514743 and third-batch7514745 launched from immutablee768b01;
independent audits7514742/44/46 depend on their actual completion. Capability
scope is explicitly enabled and disclosed as an application constraint.

Plain words: a separate typed general-reasoning boundary pilot is prepared.
Frozen local9B chooses answer or actual stronger-model delegation. Four harder
integer problems have exact enumerated CPU oracles withheld from all prompts;
packing/counting oracles independently checked by dynamic programming. Direct
local counterfactuals are recorded. Delegation executes at most four bounded
GPT-6 Astra Responses calls, with no backend substitution or fabricated answer;
another local question runs while each request is outstanding. Actual request
intervals distinguish real overlap from executor queueing. No speech, cancellation,
agent actions or training is claimed. This routing policy is explicit in the
prompt, and the seven problems are development diagnostics, not qualification.
Model and API checked in official documentation:
https://developers.openai.com/api/docs/models/gpt-6-astra and
https://developers.openai.com/api/docs/guides/reasoning .

Plain words: reasoning pilot7514755 did NOT measure model quality: the new
harness passed nested template definitions to a parser expecting flat schemas,
causing KeyError before execution. No remote calls or successful routing counted.
Correct the schema boundary and retain raw generation on format failures.
Delegate action now has no question-rewrite argument: execution sends the actual
original user problem unchanged, avoiding silent constraint changes and excessive
copying overhead. Enforce at most four requests in code. This is a client-bound
request, not a model-authored delegation brief; declare that limitation.

Plain words: full spoken/fused/scoped candidate7514743 completes6/6 reused
second-batch scenes. Arithmetic is now 'You have ten pencils left' and independent
7514744 WER0, instead of the old spoken formula WER0.266667. Most other complete
clips WER0; logic013 remains0.0909. Third-batch7514745 completes7/7 structurally,
but017 still interrupts/restarts,020 narrates East's unavailability,021 wrongly
answers the old tides request instead of two plus three, and023 asks which
project despite the explicit unavailable project. Do NOT adopt this candidate.
Actual021 ASR is exact; actual model resolve_pending_speech action consumes the
old speech ID but generates the old topic. This is a content-selection failure,
not an audio error or a missing withdrawal ID. A four-arm typed diagnostic on
that exact captured context independently varies kernels and spoken instruction;
all prompts, pending speech and source weights are otherwise matched. No training.

Plain words: typed reasoning7514992 passes7/7: three easy answers local; all four
harder direct local counterfactuals wrong, all four model-selected actual GPT-6
Astra answers exact. One foreground two-plus-three answer overlaps a real remote
request; the other three do not (their queued calls start later), so only one
proves that overlap. Routing policy is explicit; no general agent or duplex claim.

Plain words: four-arm pending diagnostic7515044 correctly rejects requested FLA
operators after loading torch-only in the same process: Transformers cached the
first module's operator choices. This is a harness isolation failure, not a
kernel quality verdict. Partial torch-arm outputs retained. Run each kernel arm
in a fresh process rather than patching module globals or weakening the check.

Plain words: isolated pending-context7515132(torch)/7515133(fused) establish
causality: both kernels WITHOUT the broad spoken instruction answer '2 plus3
is5'; both WITH it generate the old tides explanation. Same actual user context,
old pending text, assets and parameters. Reject the broad spoken instruction;
no evidence here of a kernel-caused content regression. Keep all four raw arms.

Plain words: next opt-in reasoning worker connects the local structured action
to a real bounded GPT-6 Astra request outside the media/world/local-model lock.
A direct local ordinary answer can run while that request executes. Returned
speech comes from the actual reasoner receipt, without a fabricated private read.
Only this development worker enables the external reasoner; record-worker defaults
are unchanged. Two-call per-conversation bound; no arbitrary agent actions,
reasoning-task cancellation or replacement offered in this first trial. Root
reports and independent audit distinguish local controller from external reasoner.
CPU synchronization tests verify a pending request does not block reservation and
that a cancelled task cannot publish its late receipt; these are fixtures, not
live-model proof. Actual typed worker must pass before audio; new user-only024/025
inputs get independent unchanged0.05WER/bounds quality checks and never training.

Plain words: narrower spoken-output variant now places a short natural-speech
format requirement on the function's content field only. It does not add the
rejected broad system instruction. Actual tool-result reply formatting additionally
requests only that receipt, excluding narration of other tasks/withheld notices.
This is an opt-in prompt/schema change with unchanged generated text and tool
arguments, not after-the-fact repair. Re-run the captured counterfactual first;
previous arithmetic success does not qualify this new variant.

Plain words: first actual reasoning audio7515356 completes both development
cases. User recognition024 has raw WER0.0227; ordinary ice answer then actual
GPT-6 Astra subset result both complete and independent7515358 hears both with
WER0. The subset count19 AND its size breakdown2/3/8/6 match independent CPU
enumeration. Simple9+8 speech has WER0.2, so no blanket speech acceptance. Real
request overlap already proved in typed worker7515324; do not infer actual API
overlap merely from the replay's controlled12-second delivery minimum.

Plain words: a live transport seam is prepared around the same bounded GPU
services. Actual capture frames arrive through ordered, size-limited queues;
client/server record PCM and receipt provenance. Browser worklet requests24k
and rejects another context rate; AEC is requested and applied settings recorded,
without claiming physical echo rejection. Authentication is an ephemeral first
socket message; no API keys reach the browser. Owner microphone permission is
not requested unattended. First run an actual paced PCM network test and exact
packet identity check, plus the existing independent speech audit. This remains
a development prototype, not physical microphone/device acceptance or a trained
native duplex model. Existing recorded path remains the default.

Plain words: actual network job7515423 PASSES. Independently approved024 user
PCM arrives unchanged across the socket; all249 received model PCM packets
match the server's generated packets exactly, with ordered speech IDs. Both
ordinary ice speech and actual stronger-reasoner count complete; independent
7515424 WER0 for both, with straightforward count wording rather than the prior
ambiguous number boundary. No controlled12-second delivery delay in this live
transport path. This is a paced recorded-PCM network test, NOT a physical mic,
room AEC or device-playback acceptance. Parent and child runtimes remain isolated.
Recorded default regression7515425 completes all3 cases with exact user ASR;
7515426 retains small added-word differences in008/019, and both cancellation
and owed science in020 are WER0. No native weights or optimizer steps changed.

Plain words: morning access helper is loopback-only and keeps the ephemeral
socket credential out of URLs and logs. It reads only a private owned token
file, never an API key, and denies cross-origin credential reads. Static voice
UI is proxied through an authenticated SSH forward; the socket goes directly
through that same local forward. No unattended microphone permission request.
A bounded owner-trial service wrapper is prepared; not a qualification result.

Plain words: extend the opt-in reasoning worker with actual delivery cancellation.
The model must choose identified active task IDs and its own user-facing words;
all targets validate atomically before state changes. Late stronger-model returns
remain in actual execution receipts but cannot publish. This explicitly does NOT
claim to abort an already executing remote API request. Unrelated work and owed
ordinary speech remain separate. CPU synchronization/atomicity tests pass; actual
ordinary-plus-reasoning and cancellation worker trials must pass before owner use.
A new user-only026 cancellation scene is prepared and independently judged before
any audio test. No API outputs become training data.

Plain words: actual typed ordinary7515615 preserves the pending reasoning request
and answers ice locally during its API interval. Actual typed cancellation7515616
chooses cancel_reasoning for exactly background, executes the delivery guard, and
answers the new shadow question from local parameters. The late real API receipt
is retained but deliverable=False; unrelated work is untouched. Cancellation is
explicitly delivery-only, not remote computation abortion. New026 generation
7515617 and independent7515618 both user clips WER0/bounds valid. Next actual audio
uses ZERO artificial delivery delay, so success cannot rely on holding a finished
result for the old12-second test floor. Wrapper default remains12 for old probes.

Plain words: no-delay actual cancellation7515675 completes026, live user WER0.0213.
Actual cancel_reasoning targets only turn-1; the stronger request's late receipt
is retained and discarded before speech. Only the shadow reply completes;
independent7515676 WER0. Delivery floor0, so this does not rely on an artificial
hold. The phrase 'stopped the counting task' refers to delivery cancellation;
no remote execution abortion occurred or is claimed by the actual receipt.

Plain words: browser renderer now uses one gain envelope per utterance:5ms start,
8ms interruption fade, no gain reset at native PCM packet boundaries. Received
PCM remains unchanged; browser gain is explicitly separate from server audio
artifacts. This prevents an intentional abrupt stop in the renderer's design,
not proof that model-level clicks or physical playback are solved. An offline
browser-engine check is supplied at /qa.html without mic/device output. Automated
browser QA could not run: iab/chrome providers unavailable, browser inventory
empty, native computer-use pipe startup failed. Do not bypass that with another
computer-control mechanism or claim browser/physical acceptance. Socket/auth/
private-file tests and JavaScript syntax checks pass. Owner test remains pending.

Plain words: timing audit of actual network7515423 finds an important software
barrier. Real API interval15.988–28.974s; ordinary first scheduled server PCM29.736s,
so this run did NOT speak while the API was still computing. The media loop's
bool(futures) condition holds ready foreground speech behind every worker. This
explains the delay despite verified nonblocking local inference; no model/GPU
training change can remove that client barrier. Unused owner job7515697 is
superseded before publishing access, after confirming no owner sessions exist.

Plain words: correct the barrier using validated application execution state.
Unknown semantic decisions still block queued speech. An actual lookup_record
or delegate_reasoning dispatch can continue in the background; a model plan alone
cannot clear the barrier. Retired work also cannot hold foreground speech. Read
results keep their existing revision/delivery guards and declared timing floor.
New synchronization tests exercise an actual pending reasoner fixture and a new
unresolved foreground request separately. Re-run real network inference and the
recorded regression before restarting the owner trial; require measured timing,
not just final text or reordered playback, for any concurrency improvement claim.

Plain words: corrected actual network7515717/18 verifies261 exact packets and
foreground server speech24.330s during real API interval16.020–30.120s, ZERO
artificial delivery floor. Prior run foreground first PCM29.736s after API done;
new actual-operation-aware barrier removes that software serialization. These
are server/network times, not physical device latency. Ice speech WER0. Count
text19 and its include/exclude-two breakdown9/10 are independently correct, but
ASR joins fifty/ten into5010 (raw WER0.1579): spoken-number boundary unresolved,
not accepted. Recorded3case5719/5720 preserves actual tasks; small added-word and
cancelled/canceled differences remain. No new training or native weights.

Plain words: reasoner prompt now answers a single-number request in one short
sentence without unrequested numeric breakdowns. This changes actual model output,
not scoring or postprocessing. Re-run normal and cancelled reasoning via the
actual network path; cancellation input has its own prior quality0WER gate.
Browser offline engine check remains pending because computer-use backends are
unavailable, not because microphone consent was requested. Owner service remains
unpublished until current checks; superseded5697 has no owner session.

Plain words: concurrent network probes must not assume port8788 is free on a
shared four-GPU node. Bind loopback probes to an OS-assigned port0 and report the
actual bound port in ready.json, which the client already reads. This avoids
cross-job socket collisions without changing model prompts, weights or PCM.
Owner service can retain its separately forwarded configured port.

Plain words: protect session-slot ownership in the live transport. Rejected or
unauthenticated connections must never clear another connection's busy flag.
Only the handler that actually claimed the slot releases it. The real local
socket test now opens an authenticated active session, rejects a wrong credential,
then rejects a second authenticated connection with busy1013 while the first
continues exact bidirectional PCM. No model/session was invoked by the rejected
connections. This is a protocol test, not physical microphone acceptance.

Plain words: optional compact speech actions are prepared as a latency experiment.
The model chooses native answer(content) versus clarify(content), mapping its
chosen function to the existing canonical route. Actual lookup/cancel/replace
arguments and explicit speech-ID resolution stay unchanged. No content is repaired,
no route is guessed from words, and full generated actions still validate before
execution/playback. Default protocol remains unchanged; four record/reasoning
prompt/schema/prefill views are byte-identical to source75e3883 when disabled.
Native/canonical action provenance and the option are checked in the resident
handshake. Test the same ten actual worker cases against a matched native-format
baseline before any audio adoption; token reduction alone is not qualification.

Plain words: matched resident native7516121 and compact7516122 BOTH pass10/10
actual worker contracts. Median generation stage2.116s→1.648s; median generated
tokens48→38. Sentence lengths vary, so this is practical stage latency rather
than a controlled per-token speedup. Model weights/kernels unchanged; no training.
Actual voice/transport checks remain necessary, and the published owner service
7516113 stays frozen on verified source49e820b with the original reply protocol.

Plain words: owner URL http://localhost:8767 is serving through authenticated
SSH forward8768 to nid010225:8788. Five HTTP assets match49e820b byte-for-byte;
private bootstrap matches the temporary token without printing it; private paths
404 and WebSocket upgrade works. No microphone or model session started by this
readiness check. SSH session26055 and local helper38521 are temporary. Owner
server idle/max lifetime is bounded by its4.5hour Slurm allocation. Current run
directory restricted to0700; future server roots also0700. Token600 remains
outside exports; API keys never reach the browser. Browser/device QA remains open.

Plain words: public19.33GB controller backup via SSH timed out after5.9GB; partial
files retained, never declared complete. Direct public HTTPS revision-pinned
resume runs locally with all14 expected sizes/SHA256 and no HF credentials. Do
not include temporary session credentials in any backup or readiness artifact.

Plain words: new paired typed generalization screen prepared for twelve questions:
short arithmetic, multi-digit products, sequential percentages, square root, new
subset count/assignment, quantified logic, unknown private fact, nonexistent task
cancellation and an explicit no-tools request. CPU oracles are computed only after
actual results; fraction/percentage/root checks independently verified. Every row
and failure is retained. A user prohibition blocks an external attempt before the
API call; six actual request attempts per arm bound cost. No teacher agreement,
speech or overall qualification claim. Compare frozen native/compact formats
without adapting between arms; review actual final words and receipts.


Plain words: actual compact network7516339/audit6340 and cancellation6341/42
complete with exact input and received PCM. Ordinary ice speech WER0; count19
is correct, but independent ASR changes sum to summed (WER0.167). Cancellation
preserves the new shadow answer and suppresses the late result; extra and plus
cancelled/canceled differences remain (WER0.059). Compact is not adopted in the
published owner demo7516113, which remains on49e820b.

Plain words: new typed generalization7516385 native /6387 compact finds a real
false-completion defect: BOTH claim cancellation despite no earlier task. Native
also selects external delegation against explicit no-tools wording; the harness
blocks the actual API request. Native logic is unscored because its six-request
budget was already exhausted. Native correctly answers two local and six actual
remote numerical cases; compact correctly answers four local numerical cases,
four remote, logic and unknown private fact, and obeys the no-tools product.
This is a small typed diagnostic, not a general intelligence score or speech gate.
Full actual outputs/receipts retained in recovery continuation evidence.

Plain words: targeted reasoning-domain instruction now makes actual empty task
state authoritative, disallows claiming an operation through an ordinary reply,
and explicitly honors external-tool prohibitions. No string repair, hidden
answer oracle or cancellation-by-speech is added. Recheck the original twelve
questions as development plus four new task/constraint questions in BOTH formats.
Budget raised to eight actual external requests per arm so the old six-call cap
does not hide the logic result. No live service change or optimizer steps.


Plain words: task-state16-question typed development7516456 native /6457 compact
both complete. Ten numerical outputs per arm match independent CPU oracles;
both quantified-logic answers are correct, unknown private facts are not invented,
nonexistent cancellation is no longer claimed, and both no-tools requests stay
local. Each arm chooses five real stronger-model requests. Native/compact both
add All tasks are complete when no task exists; unknown-contact wording also
makes broader unsupported privacy/storage claims. These remain wording limits;
this small typed screen does not establish complete-task or duplex qualification.
Actual existing-task cancellation still needs a regression after the prompt change.

Plain words: dispatch guard fix validates nonempty speech and every owed speech
ID BEFORE mutating a reasoning cancellation. Previously an unknown owed ID could
cancel actual work then fail validation. A fixture regression checks both invalid
ID and blank content leave the real background task uncancelled; nine targeted
worker/dispatch/format tests pass. No model-generated content is repaired.

Plain words: bounded static-cache decode experiment prepared in the component
worker only, default eager unchanged. The pinned Transformers5.5.3 source has
StaticCache layers for both linear and full attention and native generate-time
compilation. Compare eager, static-eager and static-compiled with fullgraph=True
and reduce-overhead. Record actual cache and compiled-call creation; inspect all
actual outputs, errors and warmup cost. No silent fallback is authorized, no
shared speech runtime is edited, and live/RPC integration remains unmodified.


Plain words: static eager7516499 and static compiled7516500 both FAIL0/10.
Initial hybrid mask dispatch raises KeyError linear_attention; later cache reuse
raises LinearAttentionLayer has no max_batch_size. Actual compiled wrapper/cache
creation does NOT mean successful compiled inference. Reject this runtime path;
no upstream monkeypatch, skip or fallback is adopted. Unchanged eager7516498
passes10/10, median actual controller stage2.001s. Live service remains eager.

Plain words: state-prompt/atomic guard network7516501/02 ordinary completes,
ice and actual count19 both speech WER0. Native6503/04 and compact6505/06 preserve
actual delivery cancellation and only the new shadow reply; independent ASR adds
and and cancelled/canceled spelling (WER0.0714). Exact client/server PCM in all.
Fourteen targeted frame/socket/auth/session/dispatch/reasoning/format tests pass.

Plain words: two additional approved-user-speech candidates027 nonexistent earlier
calculation cancellation and028 explicit no-external-tools multiplication24x36 are
prepared. Independent input-quality guard is unchanged; generation and judging
must use this exact source. Evaluate both formats without another prompt change,
read references only after execution, keep all outcomes under runs, never train.
These are development checks prompted by observed failure categories, not broad
fresh-system qualification or physical microphone/AEC acceptance.


Plain words: new spoken input6564 /independent6565 approves027/028 with all user
clips WER0. Native6566 and compact6568 both honestly decline nonexistent earlier
cancellation and answer24x36=864 with NO external request. Compact played audit
6569 both WER0; native6567 adds then and hears824 instead of24 as first factor,
final864 correct. No further prompt adaptation between formats, no speech repair.

Plain words: experimental owner7516574 now uses compact on tested immutable
0d63370 at nid010159:8788. Same local URL8767, new separate SSH forward8778 tool
49799, private local helper6434. Old owner6113 had zero sessions immediately
before its cancellation request; old WebSockets were not touched during HTTP
helper replacement. Five new served assets exact, bootstrap/private paths/WS
upgrade checked without mic or model session activation. Not native qualification,
physical device QA, AEC acceptance or owner voice approval. No new training.

Plain words: public Qwen backup COMPLETE:14 exact files19,329,391,706bytes, zero
errors; HTTPS fallback preserved original revision and every approved size/SHA.
Twelve speech assets6.43GB complete. Recorded87 captures/139 played clips and new
network7 captures/10 clips verified, all complete bounds valid. Gallery32 actual
conversations GET/WAV/hash checked. Keys, session tokens and actual owner audio
excluded. Runtime/source portability is still separate from destination GPU tests;
cloud validation remains paused. Morning report docs/OVERNIGHT-2026-10-18.md.


Plain words: acoustic interruption replay at the actual run_case seam reproduces
nonlexical acknowledgment restart deterministically: one controller dispatch,
two speaker.start calls for the same answer. Captured7511288 shows the same
one-question/two-generation pattern, despite Uh-huh omitted by streaming ASR.
Ranked causes are generator reset, queue release after quiet, then repetition
inside a single generation; the first two match both fixture and real event trace.
No new GPU run is needed to reproduce the software defect.

Plain words: new recorded-only acoustic-pause experiment stops causal playout at
speech onset while retaining actual TTS generator state and the exact unheard
PCM suffix. After660ms acoustic quiet with no decoded words/unresolved decision,
it resumes those samples once and continues the SAME generator. Any arrived word
commits the existing interruption path; genuine stop/replace still resolves the
old speech obligation. This is NOT a learned acknowledgment classifier. Recognized
Right may still need a semantic policy. Browser resume protocol is not implemented;
explicitly reject live use and leave published owner6574/source0d63370 unchanged.

Plain words: six exact-pause fixtures pass, including unchanged baseline restart,
prompt pause, every generated sample heard once, repeated nonlexical sounds,
actual new-word interruption and owed-ID retirement, renderer agreement through
gaps/partial packets, and aggregate-audit rejection of other answers or changed
generation provenance. Independent speech audit will keep partial fragments
unscored and score the full retained generation across its real pause, including
silence. No clipped-fragment full-text WER or removed-pause latency claim. Thirteen
pause/playout tests and fifteen worker/dispatch/format/pause checks pass locally.
Next compare unchanged acoustic and new pause on approved005, then genuine015.


Plain words: actual acknowledgment7516704 acoustic baseline restarts its answer;
7516706 pause resumes SAME generation, restores5179 unheard samples, and produces
one completed response across two playback segments. Both ASR streams omit Uh-huh.
Pause at10.414s, resume scheduling11.622s; do not equate this pause with natural
uninterrupted backchannel behavior. Actual genuine0157516708 commits its pause
when new words arrive and produces only the requested short replacement. However
its original TTS had no heard samples before user onset, so it DOES NOT establish
audible interruption latency. Test approved018 with an actually audible prefix.

Plain words: independent audits6705/6707/6709 failed in a NEW audit refactor:
word-timestamp loop reused item and overwrote the work-item dict, causing KeyError
expected. No model or speech-quality failure inferred. CPU test through actual
audit main with timestamped recognizer fixture reproduces the exact KeyError;
rename work entry separately, then all23 pause/playout/worker/format tests pass.
Rescore preserved actual WAV/packet evidence with fixed audit; do not regenerate
those conversations to hide the failed scorer. Failed audit artifacts stay retained.


Plain words: frozen-model pause ablation also needs a short-command safety check.
New029 asks for a four-sentence Moon explanation, then only Stop at14seconds.
Independent input judging remains unchanged. A no-decoded-words policy can mistake
an ASR-omitted genuine command for a nonlexical sound; explicitly test this failure
before any adoption, never label it acknowledgment intelligence. Compare unchanged
acoustic with pause, retain every result, and require a genuinely heard prefix
before treating either result as an audible interruption test. No new assistant
teacher words, training labels or optimizer steps. Current owner remains unchanged.


Plain words: fixed audits6725 baseline /6726 retained-generation005 BOTH full
speech WER0.06818; paused aggregate includes the actual silence. Decimal29.5 versus
recognized29andahalf contributes; scoring unchanged, no quality gain claimed.
Tiny interrupted baseline clip is recognized Thank you and remains partial,
unscored and not reliable evidence of its actual words.015 audit6727 finalWER0.
Audible018 baseline6728 has31926 heard prefix samples and an unwanted old-answer
restart12.821s before another acoustic stop13.374s. Pause6730 has43443 heard
prefix samples, cuts12.173s, commits on actual new words12.972s, never restarts
old generation; final first-step reply audit6731 WER0 (baseline6729 also0).

Plain words: short Stop029 input6790 /quality6791 approves BOTH user clips WER0.
Actual baseline6792 and pause6794 BOTH recognize Stop exactly, but then model
speaks the full Moon explanation again as turn-2. Complete speech audits6793/95
WER0 do NOT imply correct task behavior; this is a clear STOP failure. A supplied
reply schema currently requires spoken content even when only silence is wanted.

Plain words: explicit withdraw_speech(speech_ids) action now offered ONLY with
actual owed ordinary speech. Model must select it; no keyword router or canned
stop words. Worker validates exact known IDs and emits silence/empty content with
no external receipt. Media caller consumes those IDs, records actual silent-control
completion, and never sends empty content to TTS. Background reasoning is untouched.
Ordinary replies/cancellation still require nonempty speech. Tests first reproduced
unavailable-action rejection and actual media restart, then25 targeted tests pass.
Four old message/schema/prefill views remain byte-identical if the action is not
offered. New typed four-case actual-model probe must pass stop/withdraw, stop+new
answer, and cancel-calculation-preserve-speech before new voice regressions.
Background state in that typed probe is explicitly a fixture, never a live API.
Owner6574/source0d63370 remains unchanged; no speech-control adoption or training.


Plain words: first actual silent-action probe7516827 passes ONLY2/4. Stop-only and
explicit withdrawal correctly choose silence. Stop+first-step wrongly chooses
silence; cancel-calculation-but-finish-explanation wrongly consumes its speech ID.
Gate FAILS; no voice adoption. This exposes action-boundary regression, not a
speech/ASR problem. Targeted v2 description makes WITHOUT ANY REPLY scope explicit,
reads whole revised request, and explains that cancellation confirmation does not
fulfil an explanation. Offered only with pending speech; old views without the
new action remain byte-identical. Add three new contrast cases: shortened-answer
request, stop speech while calculation continues, cancel only calculation while
speech remains queued. All seven are typed fixture diagnostics, not live API work.
Twenty-six pause/control/playout/format tests pass, including unknown/empty/duplicate
withdrawal ID rejection. No weight update, no user-input shortcut or canned words.


Plain words: v2 actual model7516870 still FAILS: stop+first-step chooses silence,
and cancel-calculation-but-finish-explanation consumes its speech ID. New stop-
speech/preserve-calculation and cancel-only-calculation cases choose correct
boundaries. Summary case chooses the expected route/IDs but copies all FOUR old
sentences; its route-only flag is NOT semantic task success. Do not report5/7
as a general behavior score. Full outputs preserved in silent-control-v2-model-
results.json. No new voice test or owner adoption after failed gate. Next repair
must distinguish replacing an answer from fulfilling an older obligation and
background cancellation from media continuation, then inspect actual content.
No training on any of these development inputs.


Plain words: v3 repairs an overlapping model-facing contract: resolve_pending_speech
still advertised finishing/withdrawing/superseding even after a separate silent
withdrawal tool existed. Only when that new tool is offered, redefine resolve as
REPLACING the old answer with a NEW answer to the full latest revision; narrow
content-field description forbids copying the long queued explanation. Add two
abstract contrast illustrations, not actual tasks/receipts, for revised-answer
versus cancellation-with-continuation. Old views without the new tool still byte-
identical. Do not infer success from this hypothesis before the actual model gate.

Plain words: typed probe now rejects exact prior-text copies for requested new
answers and checks one sentence when explicitly requested. These are content-
contract checks, not full factual scoring; manually inspect actual answers. Three
new contrasts include Moon-topic summary, cancellation with continuing explanation,
and silence while number work continues. Ten cases total, all retained; no training
or reference words hidden from model inputs.26 local regressions pass. Live owner
source0d63370 unchanged. Proceed to voice only if all routes, IDs and requested
answer content are actually correct; do not repeat the earlier route-only claim.


Plain words: v3 actual7516958 fixes previous stop+first-step and calculation-cancel
preservation failures, and passes three new contrasts including Moon one-sentence
summary. Gate STILL FAILS: seed main-point request copies all four old sentences,
now caught by content check. Nine of ten action/content contracts pass; no full
qualification or voice adoption. Actual outputs retained. Initial submit SSH61271
ended255 without launch receipt; verified no receipt/job before idempotent retry,
which created6958. No duplicate model run was inferred from connection failure.

Plain words: observed captured018 context has consecutive USER messages and no
ongoing assistant turn; its real generated answer is only in pending-speech state.
New DIAGNOSTIC history ablation optionally inserts that same already-available
prior text as an assistant generation, while pending state still marks it not
fully heard. No new answer information or reference transcript is added. Both
arms use the same new source, weights/tools/instructions and eleven cases: ten
fixtures plus actual6730 captured ASR/first-backend/pending-state replay. Verify
actual first generated content matches pending text and saved audio hash. This
is not heard-word alignment, native speech or permission to invent assistant
history in production. Adopt a real event projection only if evidence supports it.


Plain words: matched history baseline7516980 fails seed summary as before; generated-
assistant-history7516981 passes ALL11 action/content checks, including actual6730
captured revision. Same weights/model code/tools/instructions/source for both;
only prior assistant history differs. Actual summary now condenses to one sentence,
not the four-sentence queued text. All actual answers inspected. This small typed
result supports missing generation history as a cause; not full task qualification.

Plain words: opt-in real media projection records generated text after actual TTS
start, before the next user arrives. It does not invent user words or reference
answers. Same plan resumed/restarted keeps one chronological assistant turn;
server completion updates it instead of appending a duplicate. Consumption marks
unfinished plans retired. Scalar metadata preserves already-submitted snapshots.
Generated/retired/server-complete are distinct; physical hearing ALWAYS unconfirmed.
The reasoning controller receives this delivery metadata too, even after withdrawal,
so generated text cannot silently become a fully-heard transcript. Recheck the
positive11-case model gate WITH this explicit metadata before real voice.

Plain words: 28 local pause/control/playout/format tests pass, including actual
run_case history timing, no duplicate assistant turn across repeated pauses,
withdrawal/completion lifecycle and unmutated past controller snapshot. Four full
old/new worker payloads are byte-identical when no generated-speech history exists.
New helper code is in the root source inventory. Main CLI scopes this first proof
to state-aware local9B reasoning; disabled by default and owner6574 unchanged.


Plain words: add user-only030 long seed explanation followed by the generic
main-point revision that exposed the copying failure. Independent input judge is
unchanged. Compare identical actual audio with history projection disabled/enabled;
keep explicit generated-versus-heard delivery metadata in the enabled arm. This
is development supervision/evaluation only, never training. Require eleven-case
actual delivery-aware gate7160 before voice model runs. Existing Stop029 and
revision018 use their original immutable validators. No owner service change.


Plain words: explicit delivery-state model gate7517160 also passes ALL11, so
actual voice tests proceed on unchanged model/control code. New030 user inputs
7517206/independent7207 BOTH clipsWER0. Same sourcef271aba, same initial question,
user PCM and models: baseline7208 copies four old sentences for main-point request;
history7210 gives one condensed sentence instead. Independent7209/7211 complete
speech BOTH WER0, which alone cannot distinguish the baseline task failure.

Plain words: actual bare Stop029 history7212 has correct Stop recognition, retires
turn-1 and emits silent_control_completed with NO subsequent reply or completed
speech. Its old prefix audit7213 is partial/unscored. First-step revision018
history7214 retires old turn-1 and gives only first step; independent7215 finalWER0.
Actual generated context marks old plans retired-before-full-playout and new replies
server-playout-complete; physical hearing stays unconfirmed. This supports a real
controller-context repair rather than another weight-training claim. Confirm exact
causal audio/metadata hashes in local backup before gallery/promoting. Live owner
6574 still0d63370: no pause protocol or speech-context flag has been enabled there.


## 2026-10-18 — Fresh frozen-conversation evaluation and native evidence recheck

Plain words: Sourcef09baf0 predeclares11 conversations before synthetic user
input7520271/independent quality7520272. Inference sources match demo7915b1c.
Ten eligible actual services7520283–7520301 (oddIDs) and dependent speech audits
(evenIDs) COMPLETE; all input/WAV/wire hashes and ten production-JS scheduling
replays pass locally. Memory056 fails the unchanged favourite/favorite lexical
gate and stays unscored, not regenerated. Numerics456/23 and actual Astra subset
count5 correct; ordinary foreground speech preserved. New soap-count4 cancellation
has a real delivery-cancel receipt and same-generator continuation, no number
spoken. Stop/shorten/tree-continue contracts hold. Eleven completed speech clips:
nineWER0, appointment.0435, soap.1136; allboundsvalid; sixpartials have no fullWER.
Private request057 invents no read/code but repeats the same refusal twice after
endpointing one user clip into two clauses. No overall/native/microphone acceptance.
Narrow queued-history probe reconstructs the actual second native state exactly,
then adds only the actual earlier queued controller reply in chronological history;
no model update/external request/action dispatch. Runtime remains unchanged.

Plain words: Recheck of balanced native checkpoint7510331 verifies model/state/
data/voice hashes, exact fresh reload and all16 native WAVs against independent
ASR7510355. Mechanical training passes520updates; qualification fails triangle
3,4,or5, pencil lead-is-lead, repeated clarification and no real cancellation in
either cancellation scene. Per-casep95compute36.6–44.5ms/80msframe; maxbacklog
2.83–12.12s, nine cases with causal underrun samples. Proofs frozen-conversation-
integrity.json/native-balanced-qualification-integrity.json and replayable verifiers
remain in runs/recovery-continuation-20261016; full breakdown docs/QUALIFICATION-
2026-10-18.md. Cloud paused; goal active.


## 2026-10-18 — Matched function-name training rejected; owner session lifecycle repaired

Plain words: Initial7520602 correctly fails before model loading/updates because
newer prompt-compiler source hashes differ from approved teacher7517716. Guard
unchanged. Isolatedbranch experiments/function-name-loss-20261018/source710b305
restores exact original2compilerfiles;216teacher replays+72baseline rows pass.
Matched7520623 finishes648updates; checkpoint/state hashes, optimizer/RNG restore,
allsourceparameterhashes pass.64/72, retainingall60baselinepasses. All72canonical
results/verdicts equal oldscope7517798, including wrong50 ratherthan49. No gain;
reject/noadoption/no unchangedextension. Checkpoint/state/fullresults local.

Plain words: Owner accepts recordedhybrid voice/slightfirst-response delay but
cannotusetemporarydemo. First issueSSHFwd timeout fixed. Secondissue stale session:
server rejectsnewStart whileoldsocketowns actor; disconnectedhandler alsoawaits
idle sender beforeclearingbusy. Source76174d1 adds authenticatedexplicitStart
replacement, abortablecapture/media actor, disconnected sendercancellation,
startupEnd/late-mic cleanup/completionUI reset, microphone sound/quiet indicator.
3actual localhostWS +4frame/core-abort +7Nodeclient/playout checks pass. NewGPUdemo
7520776 ready; realGPUtwozeroaudio sessions prove replacement/.879s, no microphone
orAPI. Publiclocalhost8767→newtunnel8780→nid010137:8788; exactstatic/privatebinding
checks pass. Matchedspeech7520777/audit778 preservesinput/nativewords/actions/
samegenerator, no duplicate/discardedproductionreplaysamples; completeWER.02128
validbounds. Owneraskedtoprovide Listening/sounddetected versusquiet/startfailure.
No physicaldevice/roomAEC/nativequalification. Currentvoiceweightsfrozen, nocandidate
headadopted. Reports/proofs/backups current; goalACTIVE/cloudpaused.

## 2026-10-18 — Reject normalization and queued-action proposals; bounded whole-decode diagnostic

Plain words: 7521006/b4c4928 completes20 matched calls. All source hashes/versions
unchanged, one normalization graph executed. Refrigerator word parity fails both
candidate repeats; four eligible steady speed ratios median0.995558. GateFAIL,
no gain/no adoption/no unchanged extension. Earlier historical-gate failures919/964
retained. 7521032/b1d4b00 reproduces baseline duplicate-risk fault and passes3/5:
explicit prepared reply, withdrawal, preserve-old-answer-plus-six. Original context
still creates identical refusal; replacement thirteen leaves oldturn-1 unconsumed.
All-five gateFAIL: proposed new action is not implemented. All captures synthetic,
do_not_train, zeroAPI/dispatch/updates, strict775tensor source loading.

Plain words: Next bounded diagnostic compiles the complete batch1/sequence1
forward using fullgraph=True,dynamic=True without CUDA graphs; eager prefill and
existing DynamicCache unchanged. Same-process exact raw actions/words/token counts
on five actual saved payloads/two repeats; unchanged source hashes/versions, graph
execution and zero graph breaks plus1.2 median speed ratio required. Stop at first
candidate error, preserve it, no eager fallback. This tests whole-decoder Python
launch overhead left untouched by normalization-only compilation. No live change,
new training labels or native qualification. Native balanced520 remains rejected;
its text-only temporal objectives and narrow25teacher coverage remain limitations.

Plain words: wholedecode7521255/source19bdc70 diagnostic terminates as planned:
10original exact same-process controls; first candidate fails FLA's explicit
compiler-disabled layer_norm_gated_fwd. Zero compiled graphs, no eager fallback,
sourcehashes/versions unchanged; no latency gain/adoption. ProcessCOMPLETED is
not candidatePASS. Subsequent speech-accounting proposal uses existing
resolve_pending_speech only: clarify that fulfilling a prepared unsaid answer
retires its ID, distinguish replacement versus preserving a separate obligation.
Two generic procedural examples do not contain captured user wording. Same5
fixed development probes; all required before runtime implementation; no new
function/keywordrouter/dispatch/API/optimizer, no native qualification.

Plain words: existing-resolution7521271/source23ba558 COMPLETE54s,4/5. Same-request
actualcontext and explicitprepareddelivery useexactoldanswer+turn1; stopcorrect;
replacement13 retiresoldID. Separatequestion6 explicitlypreservingoldanswer also
retiresit: wrongaction/lostwork, NOT arithmeticerror. All-fivegateFAIL/noadoption.
Newfunction7521032 remains3/5 failed, and7521006/1255 remainfailedlatencyhypotheses.
All aredo_not_train; future supervisionmustbeindependentlyauthored ratherthan
relabelledcaptures, andfreshconversationretentionisrequired afteranytraining.

## 2026-10-18 — Real microphone works in Chrome; diagnose choppy playback

Plain words: Owner still could not use earlier Listening page/no microphone line.
Fresh Chrome tab shows current page and built-in microphone permission prompt.
Owner grants browser access and confirms conversation works, but sounds choppy/slow.
Actual session metadata7520776 sessions4/5:2recognized user utterances each;6/8
retained-generator pauses/resumes and6browser underruns each. ALL14pauses have
zero new ASR tokens since preceding complete utterance. Within-event generated
packet gaps max.116/.120s. This implicates acoustic-only .003RMS pauses on noise;
it does not prove room AEC or physical hearing. No owner voice/words exported.

Plain words: New opt-in lexical-confirmed pause gates only the initial pause on
an actual alphanumeric ASR fragment +sound inpreceding1s, once per unfinisheduser
utterance. No command-keyword routing; model stillchooses resume/stop/replacement.
Defaultunchanged, clientcursor/generator/voice/endpoint/buffer/weights unchanged.
Tradeoff: genuine interruptionwaits existing.5s STTlookahead. Require exactoriginal
endpoint replay +numericprivateinputaudit, then actualnetwork Stop051/continue053/
cancel055 and speech/cursor checks before newownerdemo.11localendpoint/gate/abort
checks pass; no currentlive promotion. Capture-settings metadata nowpersisted at
sessionstart evenifbrowseraborts. Pairedsupervision authored108TRAIN/54heldout in
untrackedtraining/control/paired_*; collection/training NOTstarted during ownerfix.

Plain words: Paired-controller supervision now has108new authoredTRAIN/54heldout
fixtures, wholepair/templatefamily splits, separateitemvocabulary/numberranges.
Nine kinds include preserving/replacing/fulfilling owed speech, silentwithdrawal,
ordinarynumbers, actual retainedgeneratoraction, cancellation+resume, positivehard
delegationselection, missingcancellationidentity. No owner/capture/evaluation file
isread.5localguards execute all162 through currentnativefunctioncompiler/worker;
correctnumbers withlost/stalependinganswers explicitlyfail. Teachercollection is
separate from nativevoice training, boundedpreview9TRAINlabels/9heldoutfixtures;
preview NEVER receivesTRAINfile/approval. Allactual teacherattempts/provenance,
completeworkerstate/answerchecks and clarificationsemanticreview are retained.
Fullbatch requires every108label and exactprompt/source/response replay. No new
optimizer/trainingstarted. AnycontextualdecoderLoRA followup requires separate
recipe+realgradientpreflight+broaderretention; do not repeathead-onlypilotunchanged.
