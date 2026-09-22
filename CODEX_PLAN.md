# Codex plan — critique and proposed experiments for agent review

Date: 2026-09-21. Status: **proposal for critique, not an adopted execution plan**.

The project owner requested this standalone document so the current agent can read and challenge the adversarial review. Creating it does not change the current workflow, authorize new GPU jobs, cancel existing jobs, or supersede existing decisions. Implementation recommendations below remain proposals. The review snapshot was taken while v11 was training; later results may change the assessment.

The central recommendation is to reconsider automatic escalation from word-weight 3 to 10 until the measurement and stream-format discrepancies below have been resolved. The current evidence does not separate a model-capacity limit from a decoding choice, a data problem or an implementation mismatch.

## How to critique this proposal

For each finding F1–F5, return **agree, disagree, partly agree, or superseded**, with the exact source line, scenario, job or recomputed number supporting that judgment. Separate an observed defect from its proposed explanation. A disagreement supported by raw evidence is more useful than agreement with this document's narrative.

For each proposed experiment, identify whether it answers the stated question, whether a cheaper test exists, and whether its completion criterion could reward the wrong behavior. Report subsequent v11 results separately from the September 21 snapshot. Finish with a proposed order of work and any changes to the hour budgets; reviewing this document alone is not an instruction to execute those changes.

Use plain language. Define technical terms on first use. Paths and line numbers below refer to the reviewed repository at commit `bb53f7a`; check for subsequent changes before treating a line reference as current.

## Evidence available to the reviewing agent

- [Full adversarial review](docs/reviews/2026-09-21-phase1-audit.md): explanations, source locations and additional record corrections.
- [Captured evidence](docs/reviews/2026-09-21-phase1-evidence.json): cluster probe results, checkpoint settings, training counts, gate summaries, prompt boundaries, job accounting and source hashes.
- [Executable gate counterexample](docs/reviews/phase1_gate_counterexample.py): uses the actual repository tone, scoring and verdict functions without a model, GPU or audio-file writes. Run from the repository root with Python containing NumPy, the numerical-array library: `python -B docs/reviews/phase1_gate_counterexample.py`.
- Primary records remain `EXPERIMENTS.md`, `HANDOFF.md`, the implementation files, and cluster outputs under `/projects/u6si/duplex-live/runs/` and `/projects/u6si/duplex-live/repo/logs/`.

The review used read-only SSH and CPU checks, with no new GPU jobs. Eight checked implementation files matched between the local and cluster repositories. Probabilities in existing probe JSONs were already rounded; their original precision cannot be recovered from those files. Captured accounting is for the connected user, not an authoritative project balance.

## F1 — The gate can pass non-conversation and reward premature replies

**Observed evidence**

- Three isolated 80 ms tones, at T1=2.40 s, T2=16.64 s and T4=9.28 s, with complete silence otherwise, produce **5/5 PASS**. Total sound is 0.24 s. These are fabricated test inputs, not checkpoint results.
- T1 omits unanswered questions from its median response time (`evals/harness/metrics.py:120`). Its scored question starts are 2,17,…,137 s, but the live script plays them at 17,32,…,152 s and ends at 152 s (`evals/harness/protocol.py:19`, `:53`). The last question contributes no samples.
- T1 measures response from question start. Audible question endings, measured on the existing 80 ms grid, occur 1.52–3.60 s after onset, median 2.52 s. The median response bar is 1.5 s.
- T2 measures reply from interruption start, with an 800 ms deadline (`metrics.py:84`). The interruption audio lasts 4.80 s, with activity through 4.72 s. All ten interruption-training lessons schedule replies 240 ms after interruption end: 2,320–5,040 ms after start. The separate stopping threshold is not enforced by `evals/harness/run.py:38`.
- T3 accepts a permanently silent assistant as continuing through listener acknowledgments. T4 can accept sound during the second part of the user's request. The gate does not establish answer meaning or real-voice interruption. v2's recorded post-interruption text is `meotiveotiveer`.

**Interpretation:** Existing pass counts do not establish conversation ability. Several failure explanations based on the old latency numbers are also unsafe. Threshold values cannot be evaluated sensibly until the event they are measured from is correct.

**Proposed cheapest test/fix:** Align stimulus and scoring marks, match each answer to one question, measure stopping from interruption onset and answering from user completion, and establish preceding speech for interruption/acknowledgment tests. Check known-good scripted traces, silence, continuous noise and the three-tone counterexample on CPU.

**Completion criterion:** The counterexample fails; missing answers are counted; an assistant that waits for user completion can satisfy the intended timing bar; stopping and replying have separate results. Conversation claims additionally require meaningful text and actual voice behavior.

**Challenge requested:** Are there current checks outside these functions that prevent these false passes? If so, identify their execution path and a recorded result.

## F2 — Training, diagnostic replay and live inference have different stream contracts

**Observed evidence**

- Training puts every word piece—the smaller units into which the tokenizer splits a word—consecutively after one audio input (`layera/sampler.py:49`). Live inference inserts another audio input after each emitted piece (`layera/duplex.py:127`). **738 of 2,198 training words, 33.6%, have multiple pieces.**
- Audio preparation yields 52 features per four-second window, while the grid has 50 ticks. Training consumes concatenated features one per tick (`layera/prepare.py:86`, `sampler.py:63`); live uses all 52 with timestamps spaced by 4/52 seconds (`duplex.py:166`). Window-boundary positions differ by 160 ms per window, reaching four seconds after 25 windows.
- Live collects four seconds of audio before producing its batch of decisions, then backdates the output to audio timestamps. In v10, processing T1's 152 s takes **239.615 s**, excluding model loading. Median chunk time misleadingly reads zero because three of four incoming chunks merely buffer audio.
- `<sil>` means no new text and may coexist with continuing voice (`tokenizer/SPEC.md:29`), but the harness treats it as ending speech (`evals/harness/targets.py:159`). **9,314 of 27,429 training `<sil>` ticks occur within assistant speech spans.**

**Interpretation:** The claim that live now exactly matches training is false for the reviewed implementation. These discrepancies could contribute to failures; their individual effects have not yet been measured. Actual availability of sound to the listener is not the backdated decision timestamp. Whether audio features themselves use future information within a window remains untested.

**Proposed cheapest test/fix:** Trace one split word across a four-second boundary in every path. Record audio-feature index, audio time, word piece and actual availability time. Agree on one treatment of pieces and window boundaries, and exercise a minimal real-voice path to settle continuation versus stopping.

**Completion criterion:** The three paths produce the same intended input/output ordering and time mapping. Real elapsed-time measurements include buffering and processing. No-new-text and voice-stopped are separately defined.

**Challenge requested:** Is the rate difference explicitly compensated elsewhere? Can the current tone renderer represent the tokenizer's continuing-voice state? Supply a trace rather than a statement that the formats match.

## F3 — The exam is not a validated substitute for autonomous evaluation

**Observed evidence**

- Teacher forcing means supplying correct previous outputs; free running means feeding back the model's own outputs. The saved v2 exam has three scenarios, v3 four, and later exams eight from only interruption and timed-task classes. The sets are not uniform.
- v3's headline timing score is **3/6 positions**. v7's timing success is **1/4 positions**. Repeatedly selecting settings on these examples does not provide an independent final test.
- Diagnostic job **6762979** scores scheduled timing positions **6/6 with supplied control-token history and 0/6 with self-history**. It also emits `<t+30s>` at **159 wrong silence positions** in free mode. This contradicts the statement that free mode never emits its first timing token.
- The diagnostic's forced mode does not supply correct word pieces: word labels become `None`, and the model's own prediction is fed back even in forced mode (`layera/freerun_diag.py:141`, `:160`). It is therefore not an exact reproduction of the probe.
- In held-out `barge_in-017`, both user utterances and both assistant utterances occur somewhere in the training set. This is a new scenario combination, not wholly unfamiliar language.

**Interpretation:** v10's weak exam and mute live result are one consistent observation, not predictive validation. Calling self-history failures a non-issue is unsupported. The actual diagnostic shows missed scheduled emissions plus many mistimed emissions; it does not establish the claimed supplied-primer mechanism.

**Proposed cheapest test/fix:** Make forced replay use the actual training sequence and live output restrictions. Evaluate v2, v3, v10 and completed v11 on the same small set under both histories. Include response starts, interruptions, listener acknowledgments, background speech and scheduled callouts. Reserve unfamiliar wording and voices for final selection.

**Completion criterion:** Report the relationship between exam changes and correctly placed autonomous responses, with unwanted responses counted too. An improved exam with unchanged or worse autonomous behavior falsifies its use as the sole selection instrument. Actual recent clean gates cost 471–503 seconds each; probes cost approximately 74–84 seconds.

**Challenge requested:** What evidence beyond v10 supports replacing the gate? Does any newer paired comparison survive the format and scoring corrections?

## F4 — The apparent dose response is confounded and “exact zero” is rounded

**Observed evidence**

- `layera/probe_p1.py:144` rounds saved probabilities to three decimals. `0.000` can represent a nonnegative mean below approximately 0.0005. It does not demonstrate an exact-zero cliff.
- v5 and v6 use the same **80 distinct scenarios**, with repeated files. Actual visits over 400 updates differ:

| Recipe | Timed-task visits | Interruption visits | Other visits |
|---|---:|---:|---:|
| Standard 80-lesson run | 50 | 50 | 300 |
| v5 | 150 | 40 | 210 |
| v6 | 100 | 150 | 150 |

- v6 also enables yield weighting. Sorted, unshuffled file order changes the final class seen from timed tasks to delegation (`layera/train_p1.py:142`, `:154`). These are changes in class exposure and order, not fivefold distinct data.
- The main weighted prediction loss includes ignored audio positions in its denominator (`train_p1.py:189–203`). The rare-token penalty divides only by graded positions (`:217–220`). Their relative scaling needs an explicit interpretation.
- v8's rule ignores the correct answer in its learned direction (`train_p1.py:174`). A tiny CPU check of how the loss changes model weights could have caught this before job 6756384, which ran for 3,412 seconds.

**Interpretation:** Single-setting comparisons and bracketing can be useful, but the evidence does not establish a general “cliffs, not slopes” rule. The discrete winning token can change abruptly while underlying scores change smoothly. The penalty post-mortem also omits important normalization details.

**Proposed cheapest test/fix:** Store unrounded scores, record actual event counts, normalize both loss terms deliberately, shuffle with a recorded random seed, use the same evaluation set and repeat a reference run. Compare data balance and word weighting in a small crossed experiment—testing combinations rather than assuming they act independently.

**Completion criterion:** Each contrast changes its declared factor while holding data exposure, order policy, total updates and evaluation constant. A repeated reference result establishes whether the observed difference is reproducible.

**Challenge requested:** Can a particular cliff claim be defended from unrounded outputs or repeated runs? What exactly did “five times data” mean operationally?

## F5 — Separate decoding choices, data balance and model capacity before another dose ladder

**Observed evidence**

- Two 10 × 2,048 tables add **40,960 trainable values**. Most model weights remain frozen. The spec includes a trainable audio adapter—a small module adjusting incoming audio representations—and light voice-retraction training (`docs/09-layer-a-spec.md:107`); the reviewed trainer implements neither.
- The current 80 lessons contain **29,754 ticks**, including **27,429 `<sil>` ticks: 92.19%**. After splitting words into pieces there are **30,647 graded positions**, of which `<sil>` is **89.50%**. Backchannel lessons are 84.96% silence ticks; timed tasks 98.00%. The ratio is neither uniformly 95% nor a direct measure of acoustic silence.
- Accepting any old-vocabulary prediction as a word raises v3's score from **15/100 to at most 17/100**, and v10 from **21/240 to at most 32/240**. v10 predicts `<sil>` at **207/240** desired word positions. These upper bounds may include old special tokens because the saved error bucket groups them with ordinary words.
- Live selects the highest output score (`layera/duplex.py:141–144`). No sweep of silence bias, word-group choice or sampling is implemented there or recorded in the ledger.

**Interpretation:** Exact-word scoring penalizes valid alternatives, but wrong wording is not the dominant observed defect. Loss weighting is a reasonable hypothesis, not yet the uniquely supported solution. A frozen model may select existing capabilities through its new tokens; its ability to learn the required policy is unproved, not disproved.

**Proposed cheapest tests**

1. Test a logit bias—an adjustment to silence's raw output score—and selection based on total probability of words versus control tokens. Re-run with self-history because a changed choice affects later choices. An unconditional minimum chance of speaking can create background chatter; assess both desired and unwanted responses. These are free of training cost, not inference GPU cost.
2. After repairing the stream, deliberately teach four short conversations repeatedly. Compare the ten-row approach with a small audio adapter or small trainable matrices inside the model. Require autonomous response start, continuation, interruption and restart. This is a capacity test, not evidence of generalization.
3. Compare response-focused excerpts and broader distinct lessons against reweighting. Preserve tick timing within each excerpt and retain long-wait examples. Random deletion of silence ticks would alter the clock being taught.

**Completion criterion:** If a small adapter learns the four conversations under the same setup and the ten-row approach cannot, propose ending the ten-row-only ladder. If both fail, inspect labels and the stream before scaling. Select speaking behavior, unwanted speaking and answer meaning separately; keep exact matching as a diagnostic.

**Challenge requested:** What concrete observation would cause the current plan to abandon row-only training? What rules out a cheaper inference-time adjustment? Does a new v11 result distinguish any of these explanations?

## The three dumbest decisions, stated bluntly for rebuttal

1. Trusting a conversation gate that three beeps can pass, with response deadlines that encourage speaking before the user finishes.
2. Turning rounded zeros and repeated lessons into an experimental law without controlling exposure, order or repeatability.
3. Declaring self-history failures harmless and the exam predictive despite losing all six scheduled timing hits and contradicting the diagnostic's own output.

These criticize the reviewed decisions, not the people making them. A supported rebuttal should replace a mistaken claim in the agent's response rather than quietly accepting this account.

## Proposed use of the next 10 and 100 GPU-hours

These are spending caps for discussion, not measured runtime promises or job-launch instructions. CPU investigation and the contract decisions precede interpreting new training comparisons. M2 here means the real-voice module referred to that way in HANDOFF, not the different milestone numbering in the original spec.

| Next 10 GPU-hours | Question to answer |
|---:|---|
| 2 | Compare existing checkpoints on one corrected set under supplied history and self-history. Which is a useful starting point? |
| 2 | Sweep silence-score adjustments and word-group selection. Is the problem substantially a decoding decision? |
| 4 | Compare ten rows with a small adapter on four conversations. Can either learn the basic autonomous behavior? |
| 2 | Exercise minimal real voice and actual output availability. Can it continue, stop and keep up with incoming audio? |

| Next 100 GPU-hours, including the first 10 | Proposed allocation |
|---:|---|
| 10 | The diagnostic block above. |
| 30 | Controlled data/capacity comparisons with repeated reference runs and broader distinct lessons. |
| 25 | Real voice and serving: processing speed, bounded pending speech, interruption and restart. |
| 20 | Unseen language and voices, human-recorded inputs, modest noise, longer sessions and answer-meaning checks. Check preserved text ability when expanding trainable weights. |
| 15 | Confirm and scale the winner, or investigate the documented alternative-model route if the corrected Qwen approach cannot learn the short task. |

Voice and serving deserve attention alongside model experiments: the captured implementation already processes input slower than it arrives, and voice continuation affects the meaning of the training labels.

The **4,974 node-hours** figure is a September 14 portal snapshot (`docs/07-roadmap.md:8`), not a freshly verified remainder. At four GPUs per node, it represents **19,896 GPU-hours equivalent**. If all remained, consuming it before October 21 would require roughly 28 GPUs continuously. Maximizing consumption is not the objective; resolving the largest uncertainties before expiry is.

The connected user's returned job records total approximately **49.95 allocated GPU-hours** since September 14, including a partial v11 run: training/probes about **8.95**, rendering/filtering **20.26**, and live demos **15.34**. This is neither authoritative project billing nor proof of the remaining balance. It suggests the principal misallocation is attention and experiment sequence, rather than most compute having been spent on loss changes.

## Findings that survived scrutiny

- The v8 post-mortem correctly identifies a learning rule that ignores the correct answer.
- v3's error histograms genuinely show timing tokens replacing silence. The stronger supplied-primer story is what the evidence challenges.
- Saving both new input and output rows is correct; they are separate tables.
- Freezing the model is a sensible cheap baseline if there is a clear test for when to broaden what can learn.
- Synthetic lessons, weighted losses and a tone renderer are useful development tools within their measured scope.
- The per-test reset fix is present and appropriate.

## Suggested response from the current agent

Return a short disposition table for F1–F5, then the strongest objections to this proposal. Cite newer runs or source changes where relevant. Identify the three cheapest decisive tests, the proposed order of work, what would change your mind, and which existing commitments would need reconsideration. Keep that critique separate from execution so the project owner can assess the proposal without an implicit workflow change.
