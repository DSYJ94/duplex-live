# Adversarial Phase-1 review — 2026-09-21

The next automatic escalation from word-weight 3 to 10 is not justified. The measurement system and the training/live format disagree in ways that can explain the failures. Fix those before attributing the outcome to a particular learning rule or to insufficient model capacity.

This review read the requested sources in order, then used read-only SSH to inspect probe JSONs, checkpoint metadata, training logs, gate logs, training timelines, prompt durations and job accounting. The eight checked implementation files matched between the local and cluster repositories. Local commit: `bb53f7a`. No new GPU jobs were launched and no model or evaluation source was changed. v11 was still training in the captured records; there was no v11 probe result to review.

Raw evidence and derived counts are saved in `2026-09-21-phase1-evidence.json` beside this report. Cluster paths below start at `/projects/u6si/duplex-live/`. The executable `phase1_gate_counterexample.py` reproduces the gate failure using the actual tone, metric and verdict functions, without a model or GPU. Probe probabilities were already rounded in the original JSON; their missing precision cannot be reconstructed.

## 1. The gate can reward behavior that contradicts the training policy

**What's wrong.** The gate does not yet establish reliable turn-taking, much less conversation. I supplied one 80 ms tone in T1 at 2.40 s, one in T2 at 16.64 s, one in T4 at 9.28 s, and silence everywhere else. The actual code returned **5/5 PASS**. Total sound: **0.24 seconds**. These are fabricated test inputs, not checkpoint results.

The concrete defects are:

- T1 ignores unanswered questions when computing the median response time. One pulse can pass the ten-question test. Separately, scored question starts are `[2,17,...,137]`, but the live input starts at `[17,32,...,152]`. The tape ends at 152 s, so question ten contributes zero samples. Sources: `evals/harness/metrics.py:120`, `evals/harness/protocol.py:19`, `evals/harness/protocol.py:53`, `evals/harness/targets.py:225`.
- T1 measures from question **start**, not completion. The prompt files last **2.00–3.76 s**; checking audible activity on the same 80 ms grid puts their endings at **1.52–3.60 s**, median **2.52 s** after onset. Thus even after excluding file padding, a model that consistently listens to the entire question cannot meet the **1.5 s** median-response bar as intended.
- T2 demands a reply within **800 ms of interruption start**, although `t2_interrupt.wav` lasts **4.80 s**. All ten training interruption lessons schedule the assistant **240 ms after interruption end**; their replies start **2,320–5,040 ms** after interruption start, median **3,640 ms**. The gate's separate 800 ms stopping threshold exists but is never used by the verdict. Sources: `evals/harness/metrics.py:84`, `evals/harness/run.py:38`, `evals/harness/protocol.py:41`; evidence keys `prompt_durations_s` and `interruption_training`.
- T3 lets a wholly silent model pass “continue despite the listener saying mm-hmm”; it never establishes that the assistant was speaking. T4 accepts any suitable pulse after the second-part **start**, including while that **4.16 s** prompt is still playing. Sources: `metrics.py:98`, `metrics.py:107`.
- Text meaning, an actual revised answer, intelligibility, continuous speaking, and successful interruption of the voice generator are absent from the verdict. The optional content judge explicitly calls its verdicts advisory (`evals/harness/jev_scorer.py:3`). v3's T4 “real text” includes repeated commas and “between between”; v2's purported post-interruption answer is `meotiveotiveer` in the raw logs. Producing these strings is not answering the question.

**Cheapest fix/test.** Before another training decision, fix the stimulus/mark mismatch; measure stopping from interruption onset and replying from user completion; require each question to receive its own answer; and require actual preceding speech for interruption/backchannel tests. Check known-good scripted traces, permanent silence, continuous noise and the three-tone counterexample on a CPU. Keep tone tests as an explicitly limited decision diagnostic. Add a short transcript review and real-voice interruption test before claiming conversation competence.

## 2. Training, the diagnostic and live inference do not implement one stream contract

**What's wrong.** The claim that live now “mirrors training exactly” is false in several independently verifiable ways.

- Training places all pieces of a split word consecutively after one audio feature (`layera/sampler.py:49`). Live emits exactly one piece, then inserts another audio feature (`layera/duplex.py:127`). The tokenizer contract says pieces should span ticks (`tokenizer/SPEC.md:51`). **738 of 2,198 training words (33.6%)** split into multiple pieces: this is common, not an obscure edge case.
- Audio preparation returns **52 features per four seconds**, while the grid has **50 ticks**. Training concatenates windows and takes the first feature per grid tick (`layera/prepare.py:86`, `layera/sampler.py:63`). Live uses all 52 and assigns each **4/52 seconds** (`layera/duplex.py:166`). Each full window therefore adds two extra decisions live compared with the nominal grid. Window boundaries diverge by **160 ms per window**, or **4 s after 25 windows**. The free-run log reports **1,165 features / 1,120 ticks** and **1,341 / 1,289**, contradicting the claim that this is merely a harmless edge effect.
- Live waits for four seconds of audio, then stamps its decisions back onto that audio's earlier timestamps. Those stamps are not the times a user could hear the output (`layera/duplex.py:159`, `evals/harness/targets.py:156`). On v10, processing T1's 152 s took **239.615 s**, excluding model loading; every reported chunk median is nevertheless **0.0 s**, because three out of four input chunks just buffer audio. Source: raw `runs/harness-layera-p1-v10/t1/log.jsonl` and the four-second buffering code. This is already a systems limit, not merely future serving polish. Whether the audio encoder additionally uses future information within a window still needs a separate check.
- `<sil>` means **no new text** and can coexist with ongoing voice in the tokenizer contract (`tokenizer/SPEC.md:29`). The harness treats it as ending speech (`evals/harness/targets.py:159`). Of **27,429 training `<sil>` ticks, 9,314 occur inside assistant speech spans**. The grid has **11,629 voiced ticks**, not merely 2,198 isolated word ticks. A one-tone-per-piece renderer does not preserve that distinction.

**Cheapest fix/test.** Trace one split word across a four-second boundary, recording audio-feature index, audio time, emitted piece and actual availability time in every path. Require training, forced replay and live replay to agree. Choose one explicit treatment of word pieces and encoder window boundaries. Use the same implementation to build training and live streams. Distinguish “no new text” from “voice stopped”; exercise a minimal real-voice path now. Merely deleting silence ticks or making all lessons denser would corrupt the very clock being taught.

## 3. The exam has not been validated as a gate substitute; its own diagnostic is mislabeled

**What's wrong.** Teacher forcing means supplying correct previous answers; free running means supplying the model's own previous answers. Their difference is precisely the uncertainty that must be tested.

- The saved v2 exam contains **3 scenarios**, v3 **4**, and v4–v10 **8** from only interruption and timed-task classes. The ledger's universal “eight unseen lessons” description is incorrect. v3's headline timing success is **3 correct positions out of 6**, not evidence from a large independent sample. v7's claimed timing breakthrough is **1 of 4 positions**. The same examples were repeatedly used to choose settings. Sources: `runs/p1-v*/probe.json`; `layera/probe_p1.py:83` selects the first eligible manifest rows.
- v10's exam showing poor word output and its live run being mute is one consistent observation, not a validated predictor. v2 is stronger on exact word matching yet mute on T1; v3 produces some T1 pieces. Different exam sets make aggregate ranking still less defensible.
- The free-run diagnostic's “forced” mode does **not** feed correct word pieces. It sets a word's correct token to `None`, then feeds its own prediction even in forced mode (`layera/freerun_diag.py:141`, `:160`). It also uses one emission per grid word, unlike training's multiple pieces. The diagnostic is a partial forcing experiment, not a reproduction of the probe.
- Job **6762979** found **6/6** correct timing positions with partial forcing and **0/6** with self-history; both modes scored **0/30** desired word positions. Calling the self-history problem a “non-issue” because silence improved is unjustified.
- The same job's free mode emitted `<t+30s>` at **61 + 98 = 159 wrong silence positions**. The ledger says free running “never emits the first t+30s” and therefore never gets primed (`EXPERIMENTS.md:150`). The log directly contradicts that explanation. The observation is **missed scheduled emissions plus substantial mistimed emissions**, not an inability to emit that token without a supplied primer.
- Excluding scenario IDs does not establish unfamiliar language. In held-out `barge_in-017`, **both user utterances and both assistant utterances** occur somewhere in the 80 training lessons. This is not the same as exact scenario leakage, but it is weaker generalization than a held-out wording/template test.

**Cheapest fix/test.** Fix forced replay to use the actual training token sequence and live masks. Replay the same small, fixed set under both histories for v2, v3, v10 and v11 when complete. Include response starts, interruptions, listener acknowledgments, background speech and scheduled callouts. A setting that improves the exam but fails to improve correct live response starts—or increases background responses—falsifies its use as a selection proxy. Keep a second set of unseen wordings and voices untouched until selection is complete. The actual recent probes took **74–84 seconds**, while clean gates took **471–503 seconds**: retaining the gate costs roughly eight minutes per candidate.

## 4. The dose-response conclusions outrun the experiment design

**What's wrong.** A dose-response experiment changes a setting and measures the resulting behavior. That is useful only when the other meaningful differences are controlled.

- “0.000 EXACTLY” is not an observation of zero. `layera/probe_p1.py:144` rounds stored probabilities to three decimals. A stored zero is compatible with any nonnegative mean below roughly **0.0005**. The records cannot prove a mathematical cliff. The discrete choice of the highest-scoring token can change abruptly even if its underlying score changes smoothly.
- v5 and v6 both still trained on **80 distinct scenarios**. Their shard directories contain **120** and **160** files, respectively, representing repeated lessons. At the fixed 400-step budget, actual class exposure was:

| Run | Timed-task lesson visits | Interruption visits | Other visits |
|---|---:|---:|---:|
| v3/v4/v7/v9/v10 | 50 | 50 | 300 |
| v5 | 150 | 40 | 210 |
| v6 | 100 | 150 | 150 |

  v6 also switched on yield weighting. This is not a clean test of “five times more data at the same weight.” The trainer walks sorted filenames without shuffling (`layera/train_p1.py:142`, `:154`). Baseline runs end on timed tasks; v5/v6 end on delegation. Learning order and checkpoint position change along with class exposure. No reported repeat demonstrates that the apparent non-monotonic response is reproducible.
- The weighted prediction loss divides by weights for **all positions**, including audio positions whose losses were ignored (`layera/train_p1.py:189–203`). The extra rare-token penalty divides by **graded positions only** (`:217–220`). This changes their relative scale. At weight one, the prediction-loss normalization is approximately twice as large as a graded-position-only normalization. The docs' “1,500 penalties versus one up-push” explanation omits these normalizations and the actual probability/gradient magnitudes.
- v8's mathematical error is real: its answer-dependent term is a constant, so changing the correct label does not change the learned direction (`train_p1.py:174`). A two-example CPU derivative check could have caught it before the **3,412-second** job.

**Cheapest fix/test.** Store unrounded scores and score margins, fix or explicitly document both normalizations, shuffle with a recorded seed, and log actual counts of each training event. Compare both settings on the same exam. Repeat a reference run before interpreting a reversal. Run a small crossed comparison of training class balance and word weight, with equal total updates, instead of an open-ended sequence of increasingly large doses.

## 5. Word-weight 3 is a hypothesis; model capacity and decoding remain unseparated

**What's wrong.** The checkpoint adds **40,960 trainable values**: two tables of 10 × 2,048 values. That may be enough to select existing skills, but it does not establish that a frozen model can learn this unfamiliar interleaving, interruption policy and timing system. The spec originally included a trainable audio adapter and light voice-retraction training (`docs/09-layer-a-spec.md:107`); the current trainer implements neither. Frozen old word rows cannot directly learn new word outputs, although gradients through the new history embeddings can still alter later internal states. Failure is not proof of impossibility; success is not established by silence accuracy either.

The “95% of every lesson is silence” explanation is numerically and semantically wrong. Rebuilding the current 80 training lessons gives:

| Quantity | Recomputed value |
|---|---:|
| Grid ticks | 29,754 |
| `<sil>` ticks | 27,429 (**92.19%**) |
| Word emissions / word pieces | 2,198 / 3,091 |
| Graded positions after splitting words | 30,647 |
| `<sil>` share of graded positions | **89.50%** |
| Silence share in backchannel lessons / timed tasks | **84.96% / 98.00%** |

There is a substantial imbalance, but it is not uniform and many `<sil>` ticks mean continuing voice. Word-heavy training crops, broader data and reweighting are all reasonable experiments after the contract is fixed. Preserve elapsed time and long-wait examples rather than removing ticks indiscriminately.

The exact-word score does penalize valid alternative wording. However, accepting any old-vocabulary prediction as a word raises v3 from **15/100 to at most 17/100**, and v10 from **21/240 to at most 32/240**. v10 predicts `<sil>` at **207/240** desired word positions. These bounds come from the saved error histograms; their old-vocabulary bucket can include old special tokens. Wrong wording is therefore not the main observed defect. Use separate scores for speaking at the right moment, unwanted speaking, and answer meaning; retain exact matching as a diagnostic rather than the primary speaking score.

The current live decoder simply chooses the highest score (`layera/duplex.py:141–144`). There is no silence-bias sweep, word-group threshold or sampling control, and no such experiment is recorded in the ledger. Before retraining, vary a **logit bias**—an adjustment to the model's raw score for silence—or compare the total probability of ordinary words with that of control tokens, then choose a word within the selected group. Re-run with self-history because changing one choice changes later behavior. An unconditional minimum chance of speaking can create unwanted responses; judge the tradeoff on response and background cases together. These tests cost **zero training hours**, not zero inference GPU time.

**Cheapest decisive capacity test.** After stream repairs, deliberately overfit four short conversations: teach them repeatedly, then require autonomous response starts, continuation, interruption and restart from a clean state. Compare the existing ten-row method with a small trainable audio adapter or small internal update matrices. If the latter learns these behaviors and the ten-row method cannot under the same setup, stop the ten-row-only dose ladder. If both fail, inspect the contract and labels before buying larger runs.

## The three dumbest things we did

1. **Trusted a conversation gate that three beeps can pass**, while its response deadlines reward talking before the user finishes.
2. **Promoted rounded zeros and repeated lessons into a “cliffs, not slopes” law**, without controlling lesson order, exposure counts or repeated trials.
3. **Declared self-history failures harmless and the exam predictive**, although the diagnostic lost all six scheduled timing hits and its logs contradicted the supplied-primer story.

## Allocation: next 10 GPU-hours

CPU work comes first: repair the gate, establish one stream contract, fix diagnostic forcing and loss normalization, and preserve the counterexample as a guard against future false passes. The following are budget caps, not claimed measured runtimes:

| GPU-hours | Work and decision |
|---:|---|
| 2 | Re-evaluate v2/v3/v10/v11 on one corrected small set with both histories; cache detailed scores. Establish whether any checkpoint is a useful starting point. |
| 2 | Sweep silence score adjustments and word-group selection on promising checkpoints. Select on response/background tradeoff, not speech volume alone. |
| 4 | Paired four-conversation capacity test: ten rows versus a small adapter, including autonomous evaluation. Stop escalation if there is no basic learnability. |
| 2 | Connect a minimal real voice and measure actual output availability, sustained processing speed, queued voice and interruption stopping. |

Do not automatically turn a flat v11 exam into word-weight 10. Make the next training purchase answer a different, falsifiable question.

## Allocation: next 100 GPU-hours, including those first 10

| GPU-hours | Work and decision |
|---:|---|
| 10 | The diagnostic block above. |
| 30 | Controlled capacity/data comparison with repeated reference runs: broader distinct lessons, response-focused crops and minimal adapters. Expand only approaches that pass autonomous short-conversation tests. |
| 25 | Voice integration and serving: sustained processing fast enough to keep up, bounded voice queues, interruption/restart, and real elapsed-time evaluation. Start this alongside model investigation. |
| 20 | Broader evaluation with unseen text templates, voices, modest noise, some human-recorded inputs, longer sessions and semantic answer checks. Check preserved text ability when expanding trainable weights. |
| 15 | Scale and confirm the winner, or exercise the already documented alternative-model route if no corrected Qwen experiment learns the short task. |

The budget itself needs a fresh authoritative balance. **4,974 node-hours** is a September 14 portal snapshot, not a September 21 accounting result (`docs/07-roadmap.md:8`). Four GPUs per node makes that **19,896 GPU-hours equivalent**, not 4,974. If all remained, using it before October 21 would require about **28 GPUs continuously**. Burning it is not the goal; removing uncertainty before the deadline is.

The connected user's returned job records since September 14 sum to about **49.95 allocated GPU-hours**, including a partial v11 run. Training and probes account for about **8.95**, rendering/filtering **20.26**, and the two live demos **15.34**. This is not authoritative project billing or remaining balance, but it does contradict a literal claim that spending to date was mostly loss experiments. The misallocation is mainly attention: serial recipe tuning while measurement, real voice and processing speed remain unresolved.

## Things that are fine

- The v8 post-mortem correctly identifies an answer-blind objective. The v3 error histograms also genuinely show timing tokens replacing silence. Those observations stand; the stronger causal stories do not.
- Saving both new input rows and output rows is correct: they are separate tables. Checkpoint metadata confirms two 10 × 2,048 tensors. Freezing the rest is a sensible cheap baseline, provided it has an explicit capacity exit test.
- Synthetic data, weighted losses and a tone renderer are reasonable development tools. None independently establishes human conversation performance. Use them within their measured scope.
- The reset fix is present in both the harness and thinker; fresh test sessions are the right design. It did not repair the other scoring/format defects identified here.
- I checked a possible optimizer issue rather than asserting it: the installed optimizer has nonzero default weight shrinkage despite masked gradients, but a small CPU check in the actual low-precision number format showed unchanged zero-gradient rows. There is insufficient evidence to blame this for the run outcomes. Explicitly disabling shrinkage for frozen rows would remove dependence on rounding if precision changes.

## Additional record corrections

- Raw live logs contain **70 total v3 word beats**, versus the ledger's approximately 45, and **8 v2 word beats**, versus 25. v2 has **3 T5 word beats**, not 10. Source: sums of `out_samples / 1920` in each per-test log. Timing tokens are counted separately.
- The ledger's v5 probe job **6751180** failed; **6751374** completed. The cited v6 probe **6754413** is absent from the returned job list; **6751789** completed after v6's training job. Preserve result-file/job provenance explicitly.
- v1/v2 log directories contain appended or copied histories; checkpoint arguments and `steps` identify their saved state more reliably than raw log length. Historical v3 arguments do not contain the formerly hard-coded timing weight, so checkpoint arguments alone do not record every historical setting.
