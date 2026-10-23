# Native model failures: evidence and open causes — 23 October 2026

There is no adopted production candidate. The latest completed training stage is
native7590052: eight new updates carrying native520 Adam clocks520→528. The new
control40 repair has not made an optimizer update. Its full numerical gate passed;
its independent BF16 gate is running. Passing numerical/gradient checks establishes
implementation readiness, not answer correctness or interaction quality.

## What the native behavioral evaluation actually asks

[The fixed ten-case development plan](../evals/fixtures/native-v2-development-20261019.json)
reuses existing synthetic TRAIN/development stimuli; it is not sealed fresh
qualification or a broad reasoning benchmark. The same user PCM is replayed to
native520 and the candidate with native speech generation, without forcing the
answer, response onset or request token. Each model runs all10 cases, followed by
independent recognition/word-clock/audio checks of all20 outputs. A second runtime
comparison repeats the complete paired20, rather than selecting passing cases.

| Cases | Actual user prompts / task | Pass condition |
| --- | --- | --- |
| 0–3 | “Why does ice float in water?”; “How is sound produced?”; “Why do plants need light?”; “Why does metal feel cold?” | Correct ordinary explanation and intelligible independently recognized speech |
| 4 | “Do not speak until I say ready. Listen quietly.” Then “Ready. What does a timer measure?” | No speech or acknowledgment before release; correct timer answer afterward |
| 5 | “Say only the first step now: put the book on the shelf. After I say go, say close the cupboard.” Then “Pause. Stay quiet until I say go.” Then “Go.” | Say first step once, stay quiet during pause, say remaining step without restart |
| 6–7 | “Please check the room marker for project Willow North.” Then “While you're checking, why do I see lightning before hearing thunder?” | Actual lookup, independent ordinary answer while pending, actual returned value spoken. Same user audio with yellow-circle versus copper-diamond receipts tests result dependence |
| 8 | “What's the room marker?” Then “Project Willow West.” Then “While you're checking, what does evaporation mean?” | Ask for project before lookup, execute actual named lookup, use brown-square receipt and answer independent question |
| 9 | “Please check the room marker for project Alder West.” Then “While you're checking, why do I see lightning before hearing thunder?” Then “Cancel that lookup.” | Actual cancellation receipt and no stale result delivery |

These task-lifecycle tests primarily concern synthetic record retrieval. They do
not establish that the voice model knows when to delegate arbitrary hard reasoning.
The32-scene reasoning/lifecycle source proposal remains unrecorded/untrained;
microturn/backchannel teaching also remains incomplete. The40 current numerical
cases concern TRAIN control histories, not40 independent reasoning questions.

## Measured outcomes

[Current A100 default-runtime review](NATIVE-PAIRED20-DEFAULT-RUNTIME-REVIEW-2026-10-23.json):
native520 has2/4 correct ordinary cores; native7590052 has0/4. Both fail the quiet,
pause and complete asynchronous workflows. Candidate answers include closely
packed ice molecules, fabricated velum anatomy, plants needing light to “see”, and
an incoherent metal/fire/skin explanation. Independent output speech recognition
matches those wrong claims closely: ordinary-case WER0.0,0.0,0.0588,0.0448. The
errors are real generated content, not merely a misleading output transcription.
This does not independently isolate every possible input-understanding error.

[Archived-TF32 comparison](NATIVE-PAIRED20-ARCHIVE-RUNTIME-REVIEW-2026-10-23.json):
each model scores1/4. Runtime/decoding sensitivity exists; toggling that policy
alone does not recover adequate behavior. Four questions are a regression check,
not a reliable estimate of general accuracy, and multiple-seed confidence
intervals or a representative fresh conversation benchmark are not established.

[Earlier controlled ancestor comparison](NATIVE-ANCESTOR12-COMPLETE-REVIEW-2026-10-20.json):
published base4/4, native5204/4, an earlier eight-update audio49 child0/4 under the
same wrapper/prompt/user PCM/voice/assets/runtime/sampler. This identifies a
training-induced weight regression in that experiment. It does not identify the
specific damaging loss, or transfer the old4/4 score to the current A100 runtime.
The earlier child is distinct from latest7590052.

[Earlier persona-only diagnostic](NATIVE-PERSONA-COMPLETE10-REVIEW-2026-10-20.json)
completed all10 with2/4 ordinary cores and remained unqualified. Prompt changes
alone have already failed to solve the whole problem on that earlier candidate.

## Training data, batch and optimization

[Saved actual recipe](NATIVE-7590052-RECIPE-2026-10-21.json) and original report:
`runs/cloud-export-20261016/tree/runs/context-training-proof-7590052/report.json`.

-49 synthetic teacher histories:27 private lookup,12 simple,5 clarification,5
cancellation. Their real conversation timelines sum to30.656 minutes, including
listening/pauses; this is not30.656 minutes of unique assistant speech. The simple
set includes the keyword-wait and pause/resume scripts. This is a narrow synthetic
curriculum, not broad conversational/reasoning training.
-The old28 actual-history components remain distinct from49 teacher histories.
“154” counts retained loss contributions, not154 unique conversations. Membership
per optimizer step is28 actual,4 reference,10 word anchors,14 clock anchors,49
acoustic teachers,25 old teacher-text and24 new teacher-text contributions.
-Physical microbatch is one conversation. Gradients accumulate the full retained
154 contribution set before each optimizer update; it is not one-random-example
SGD. The supplementary native-call text objective exposes all49 once across8
updates. The new control40 plan adds5 control histories per update, all40 once.
-Only1,894,400 rank8 adapter parameters in26 FP32 tensors are trained; base weights
are frozen. Last4 temporal layers and text head are targeted. No controlled rank
or layer-coverage sweep establishes this as an optimum.
-Latest stage:8 updates, effective LR2e-7, AdamW, clip norm1, KL1/hidden-MSE10,
word/clock anchors64. This follows an earlier higher-rate attempt; it is not proof
that the lower rate or the combined objectives solve the regression.
-One retained text-loss component decreases only0.5719308→0.5718392 from update1
to8; old-teacher CE0.4547952→0.4545460. These are component losses, not a complete
objective/convergence curve. The model has earlier training history; eight refers
to the latest stage, not its entire lifetime.

## Ranked hypotheses and tests

1. **Objective interference / learned-policy regression.** Earlier matched
weights demonstrate regression, but the responsible component is unresolved.
Prediction: zero-update per-group gradient comparisons reveal conflicting
ordinary-retention versus acoustic/control directions. Measure those contributions
without dropping any required objective from the current pilot.
2. **Coverage and domain bias.**49 short synthetic histories heavily favor record
workflows. Prediction: direct/delegate and quiet decisions break under new wording,
cues or tasks despite good teacher-forced fit. The new40 control teachers address
one coverage gap; they do not establish broad reasoning or microturn behavior.
3. **Prompt-induced over-questioning.** The current persona ends “Ask which project
if the user has not identified it.” It is used even for ordinary questions/quiet
instructions. Prediction: a matched scoped/neutral-persona diagnostic reduces
irrelevant project questions. Older persona results show this cannot be assumed
to fix the full problem; training and approved histories remain unchanged.
4. **Insufficient or poorly balanced optimization.** Few conservative updates and
multiple strong retention terms may limit learning. Prediction: component losses,
gradient directions and independently measured behavior move coherently in a
predeclared bounded schedule. Tiny loss changes alone cannot distinguish underfit,
objective conflict or a local plateau. No unauthorized extension follows.
5. **Runtime sensitivity.** Paired default/TF32 results vary. Reproduce the
base/native520/child comparison on the current pinned A100 setup and repeated
seeds before attributing differences to architecture or dataset alone.

Batch size has not been isolated as the cause. Larger physical batches would not
supply missing task coverage. We have no evidence that unfreezing the whole model
would fix these failures. A broad causal diagnosis and useful generalization are
still unproven. Existing recorded differential loops supply the observed failures;
new native reproductions/ablations await the currently owned GPU and are not
claimed completed by this review. No model source or training objective changed.
