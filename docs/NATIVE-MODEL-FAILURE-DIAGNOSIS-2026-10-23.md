# Native model failures: evidence and open causes — 23 October 2026

There is no adopted production candidate. The latest completed training stage is
native7590052: eight new updates carrying native520 Adam clocks520→528. The new
control40 repair has not made an optimizer update. Its full numerical and independent40-case BF16 gradient gates passed;
combined retained-objective readiness is running. Passing numerical/gradient checks establishes
implementation readiness, not answer correctness or interaction quality.

## Concrete repair and what remains unproven — 23 October17:10UTC

There is no single demonstrated cause of the latest ordinary-answer regression.
Narrow training coverage is established; whether loss interference, inherited
optimization, prompt bias or limited adapter coverage causes that regression
remains unresolved. Batch size has not been isolated as a cause. A longer run,
larger batch or unfreezing the base is not an evidence-based fix at this point.

| Failure | Concrete next intervention | Evidence required |
| --- | --- | --- |
| Speaking before a cue; pause/resume mistakes | Current40 control histories explicitly teach silence, release, distracting cues, pause/resume and replacement cues, preserving all existing objectives | Full current gate chain, one8-update pilot and complete paired behavior/speech tests; silence means no audible acknowledgment |
| Unreliable hard-reasoning delegation | New32-scene matched direct/basic, delegated, cancellation and replacement curriculum; actual asynchronous teacher requests and returned content | Source-v3 all40 recordings now pass; teacher capture, semantic review, native conversion and training remain outstanding. Unnecessary delegation on direct/basic cases must fail |
| Wrong ordinary answers after training | Matched published-base/parent/child comparison on the current runtime; separate scoped-persona diagnostic; exact current objective-group gradients and inherited Adam direction if pilot fails | Same audio/voice/runtime/sampling and all cases, with independent speech review. Historical gradient conflict and older ancestor regression do not isolate the latest damaging objective |
| Stale/cancelled worker results | Backend work runs outside lifecycle lock; cancellation retires delivery; replacement keeps full arrived user history; late raw responses remain preserved |11 offline lifecycle/transport tests pass; actual native request/capture/wire/audio evidence still required |

The actual GPU check at17:10:17UTC is35/40 gradient cases with0 optimizer updates;
numerical40 already passed. The latest completed training is still7590052's eight
updates, which remains unadopted. No performance improvement is claimed yet.

Mac low-power mode is0, relevant local services have nice0, and no thermal or
performance warning is recorded. No OS CPU priority/affinity/power cap was set.
The real fixed-clip4/8/10-thread benchmark selected8:4.935s versus5.199s and5.233s,
about5% faster than4 on that clip. New source audit used8 and completed; no current
source ASR process remains. These observations do not prove that hardware never
thermally throttles or establish full-corpus speed. See
[CPU measurements](NATIVE-LOCAL-CPU-BENCHMARK-2026-10-23.json) and
[complete source qualification](NATIVE-REASONING32-SOURCE-QUALIFICATION-2026-10-23.json).

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
The32-scene reasoning/lifecycle source is recorded and source-qualified, but its
actual asynchronous teachers remain uncollected and it remains untrained;
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

- 49 synthetic teacher histories:27 private lookup,12 simple,5 clarification,5
cancellation. Their real conversation timelines sum to30.656 minutes, including
listening/pauses; this is not30.656 minutes of unique assistant speech. The simple
set includes the keyword-wait and pause/resume scripts. This is a narrow synthetic
curriculum, not broad conversational/reasoning training.
- The old28 actual-history components remain distinct from49 teacher histories.
“154” counts retained loss contributions, not154 unique conversations. Membership
per optimizer step is28 actual,4 reference,10 word anchors,14 clock anchors,49
acoustic teachers,25 old teacher-text and24 new teacher-text contributions.
- Physical microbatch is one conversation. Gradients accumulate the full retained
154 contribution set before each optimizer update; it is not one-random-example
SGD. The supplementary native-call text objective exposes all49 once across8
updates. The new control40 plan adds5 control histories per update, all40 once.
- Only1,894,400 rank8 adapter parameters in26 FP32 tensors are trained; base weights
are frozen. Last4 temporal layers and text head are targeted. No controlled rank
or layer-coverage sweep establishes this as an optimum.
- Latest stage:8 updates, effective LR2e-7, AdamW, clip norm1, KL1/hidden-MSE10,
word/clock anchors64. This follows an earlier higher-rate attempt; it is not proof
that the lower rate or the combined objectives solve the regression.
- One retained text-loss component decreases only0.5719308→0.5718392 from update1
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


## Saved weight and optimizer measurements — 23 October14:35UTC

A [CPU-only saved-file audit](NATIVE-7590052-ADAPTER-DELTA-AUDIT-2026-10-23.json)
rehashes both adapter checkpoints and both full training-state files against their
original proof reports, compares the saved state copies with every safetensor,
and reads all26 Adam clocks/moments. It performs no model inference, GPU work,
API call, optimizer update or checkpoint modification. Three mathematical tests
compare its matrix-free norms with independent dense products, including a
small-delta case and an exact no-change control.

| Measurement | Actual saved-file result | Scope |
| --- | --- | --- |
| Adapter ownership |26 FP32 tensors,1,894,400 elements,13 rank8 modules,scale2,last4 temporal layers plus text head | Confirms actual stored topology; no optimal-rank/layer claim |
| Changed tensors/elements |26/26;1,894,105/1,894,400 elements | The failed pilot was not a no-op or an accidentally frozen adapter |
| Global adapter parameter change |L2norm0.0012285421 versus parentnorm7.7299355;relative0.0158933% | Relative to the saved adapter parameters, not7B base weights |
| Effective per-module LoRA matrix change |0.0299165%–0.0489651% of the corresponding parent LoRA component | FP64 rank-sized Gram computation; no full dense base matrix |
| Text-head LoRA component change |0.0299165% | Does not isolate the text head as the behavioral cause |
| Actual Adam parameter clocks |All26 move520→528 | Candidate stagecounter8 is separate from inherited Adam clock528 |
| Saved optimizer LR |Parent1e-5;candidate2e-7 | Actual saved group values; no retrospective recipe change |
| Saved frozen-parameter hash records |Equal between parent and candidate | This audit did not rehash all frozen7B tensors |

With inherited Adam betas0.9/0.999, after eight updates the algebraic coefficients
on the old first/second moments are0.43046721 and0.992027944. These are not fractions
of update influence. Measured across all26 tensors, the inherited first-moment
contribution norm is0.07920 versus0.49261 for the new-step residual; the latter is
larger. The inherited second-moment contribution norm is0.0309432 versus0.00048719
for its new-step residual, a63.51:1 norm ratio. Residual calculations include saved
FP32 rounding. This supports a cautious, history-dependent small-step experiment;
it does not prove a defective optimizer, insufficient convergence, a batch-size
cause or that resetting Adam would fix the model. Retained Adam/RNG state remains
required and unchanged.

The effective matrices are algebraic FP64 measurements, not measured BF16
activation or logit differences. Small parameter changes can alter an
autoregressive sampled trajectory. These measurements therefore do not dismiss the observed errors, prove their cause, or
replace matched native behavioral comparisons. If the new control40 pilot fails,
the next causal investigation should measure objective-group gradient directions
and repeated matched behavior, rather than infer the offending loss from final
aggregate loss or optimizer moments alone. No unchanged extension is authorized.


## Partial teacher-local gradient geometry — 23 October15:12UTC

The [first20 current control40 cases](NATIVE-CONTROL40-GRADIENT-GEOMETRY-PROGRESS20-2026-10-23.json)
supply text-gradient norm a, combined-gradient norm b, and norm d of the actual
FP32 acoustic accumulation increment. Its text-head delta is exactly zero, so
cosine=(b²−a²−d²)/(2ad) describes these two local vectors. The calculation uses
all recorded cases from an independently hash-verified backup; two tests compare
the identity with direct vectors and prevent partial evidence being an approval.

Median acoustic-increment/text norm ratio is4.1822% (range1.3340%–10.5726%).
Cosines range−0.05247 to+0.18885, median+0.04097; six of20 are negative. Local
directions are generally near perpendicular, with weak opposition in those six.
This does not show a dominant acoustic gradient on these current20 examples.

This is PARTIAL diagnostic evidence:8 no-release,8 release-reason,4 distractor-release,
with pause/resume and cue-replacement cases still absent. The complete40 gate is
not approved. It does not measure gradient sums across examples, the original154
or old49 objectives, Adam preconditioning, behavioral correctness or the cause of
the earlier7590052 regression. No additional GPU call, loss change or optimizer
update occurs. Preserve the remaining whole-corpus gates and current pilot.


## Full retained-loss accounting and historical gradient scope — 23 October

The [new CPU-only accounting](NATIVE-7590052-RETAINED-LOSS-AUDIT-2026-10-23.json)
binds original7590052 report SHA397a259d124948dec3aeed63564ad24dbf2f13bb2b3699565f9f1974e60fa3eb.
All eight records preserve all154 component identities and weights. Weighted
products/sums are independently recomputed with decimal arithmetic; four offline
tests reject missing steps, changed membership, missing rotating exposure,
changed KL/hidden formulas, changed initial criteria and incomplete final metrics.
This does not execute the model or change the running pilot.

| Weighted retained group | Before update1 | Before update8 |
| --- | ---: | ---: |
| Original28 actual text |0.571930776|0.571839228|
| Four reference KL +10 hidden |0.180896606|0.180866666|
| Old25 teacher text |0.454795210|0.454545964|
| New24 teacher text |0.760396476|0.760218150|
| Word anchors, total weight64 |0|0.067804090|
| Clock anchors, total weight64 |0|0.048533420|
| Audio49 |0.586704992|0.586618460|
| Retained154 sum |2.554724060|2.670425979|

The scalar increase comes mostly from the word/clock anchor penalties; their
zero initial value is expected because they compare with the warm parent.
Scalar magnitudes do not determine gradient direction or the cause of wrong
answers. These records were calculated BEFORE the corresponding optimizer call,
then logged afterward: row8 measures seven new updates, not the final eighth.
The rotating native49 call objective uses different examples at each update;
its per-step totals are not a same-example convergence curve. The complete
post-update8 combined154/native49 objective was not recorded.

There is a genuine final measurement for the original28 actual-history component:
its same weighted teacher-forced loss changes0.571930776→0.571255577 after all
eight updates. This is a small training-set improvement, not held-out convergence
or an explanation for the failed spoken results. All eight recorded aggregate
gradient norms exceed the unchanged unit clip bound (range5.83018–8.75904).
That confirms clipping occurs; it does not identify clipping or batch size as
the cause, and does not authorize changing a weight or threshold.

A [historical154 gradient diagnostic already exists](NATIVE-GRADIENT154-REVIEW-2026-10-20.json),
with raw archive SHA d8a379672e4ad19d59762fa138d4c7bd6caf937d5b4898e785068987ed8e5f83.
It measured all154 individual components at the frozen native520 parent and
compared directions for its then-current first/mean-eight schedule. Some
directions opposed one another (actual/reference cosine−0.62790 in its mean
schedule), supporting objective interference as a hypothesis. Its exposure
weights predate the latest complete154-at-every-update plus recorded-call49
auxiliary, and it contains no new control40 objective or sequential candidate
measurement. It is historical evidence, not a causal isolation of7590052 or the
running pilot. Do not repeat an identical old probe or report all gradient
analysis as absent. If the new pilot fails, a useful next diagnostic must cover
its exact current complete154/native49/control40 weights and actual inherited
Adam direction, with matched behavior; preserve every required objective.


## Complete40 gradients and new teacher failures —23 October18:39UTC

All40 control gradients now pass independent readback, with0new optimizer updates.
The full local text/acoustic geometry has9negative dot products, median cosine
0.03519363 and median acoustic/text norm ratio3.9915%. This remains teacher-local
evidence; it does not isolate the original154/native49/Adam conflict or establish
the cause of7590052's answer regression. The exact retained-objective readiness
child is still running, preserved under a22:30UTC supervisor deadline.

The separate new reasoning teacher corpus stopped at6/32: cancellation lacked
a state-changing request and delivered stale content; replacement computed the
correct new result but exposed an early-close recorder bug. Both diagnoses are
bound in the teacher-failure receipt. They explain rejection of these NEW
teachers, which postdate7590052; they cannot explain the old model regression.
The new completion/ack-aware recorder passes its regression tests but has not
collected a repaired teacher. All32 independent quality/semantic/native gates
remain required. No additional current native training or behavior success is
claimed by teacher or numerical transport tests.


## Actual cancellation-state race —23 October19:13UTC

The v2 reasoning teacher selected a real cancellation request before the old
result was delivered: backend start49.620s, old wire54.996s, completed cancellation
59.747s. The application published the old result while that newer state decision
was still executing, then correctly rejected the now-late cancel action. This
is a demonstrated application ordering defect in the new teacher workflow.
A versioned state barrier holds older delivery while later native-selected
reserved/allowed-target decisions are unresolved; unrelated completed decisions
release it and uncertain failures stay closed. The old and repaired paths have
a reproducing integration test. Native v3 acceptance is still outstanding.

The replacement timing fix DID pass an actual teacher capture with the full
corrected response in959 prefix/live/span frames. Independent whole32 audio and
spoken semantics remain required. Both findings postdate the historical7590052
model and cannot establish its earlier answer-regression cause. C9 GPU training
code/objectives, stored histories and user-facing native runtime remain unchanged.


## Current repair boundary —23 October19:38UTC

[V3 actual results](NATIVE-REASONING32-V3-RESULTS-2026-10-23.json) now show an actual
cancellation receipt, no stale original delivery and a963-frame episode. The
slow pending-decision barrier is regression-tested, but that particular slow
interleaving did not occur in this actual capture; whole32 remains unapproved.
The next scene completed transport but measured1125frames against1100, so capture
stopped. Its45.75s source is unclipped and word-qualified. A new same-words/voice
provider take at a brisk natural pace is a proposed timing repair, not yet a
proven fix for the full variable-latency episode. Preserve all old bytes/receipts,
explicitly record the changed request instructions, and reapprove all40 source
objects in a new derived version. No crop/gain/window change or passing subset.
All289 raw teacher/history files are independently SHA256-verified on the approved
persistent volume. No new source take or optimizer update occurred in this turn.

At19:36–19:37UTC the existing readiness child41540 is still running;11/49 complete
teacher rows are logged, all with exact native forward hashes. All49 and complete
combined154/native49/control40 readiness remain required before the bounded8.
The numerical40 and gradient40 gates passed earlier. No reliable completion ETA.

[Fresh CPU readback](NATIVE-CPU-RESOURCE-READBACK-2026-10-23.json) separates OS
settings from application threads: Mac Low Power Mode0, nice0, no recorded
thermal/performance warning. ASR8 is the fastest of tested4/8/10 on one short
clip; no current ASR is running. The GPU process ALSO has OMP_NUM_THREADS=8,19
threads and affinity0–254. This application setting is not proven optimal; do not
claim every available CPU is used or that hardware never throttles. Background
OneDrive/FileProvider/indexing is active. Do not restart immutable readiness to
change threads; benchmark a matched workload before a future launch change.


## Actual data repairs and remaining native-request failure —23 October20:22UTC

The diagnosed source was rerecorded once with identical approved words/model/echo
voice and an explicit new pacing instruction/hash. Duration45.75→39.85s,peak0.6713,
independent WER0.0. Complete40 source objects reapproved in V4, preserving39
unchanged WAV/ASR artifacts and reassessing every verdict. Source SHA
3bccef7ba53835efc6c5be8b5d490551a2ad31bd5d3875d1cf0692e2d9f0254c;quality SHA
e9f0296a5fcddd892556bfa409ef692b5f51956c76335f1d24e7b4691781ca0d. The actual
timing teacher now fits1055/1100frames. No audio edits or threshold changes.

V4 then exposed a different replacement failure: the backend computed the changed
answer but chose an independent response, leaving the obsolete request deliverable.
Actual output combined new item names with old16-token/32-point totals; independent
ASR confirmed those totals in the unclipped original WAV. A clarified generic
model-selected action contract passed one actual backend-only replay and independent
exact state/receipt recomputation. Its original whole-voice checker remains failed
because the component deliberately sent no voice commentary; no API retry occurred.
V5 then passed an actual native teacher replacement in992frames, retiring the old
request and delivering only the current receipt. This is component/teacher evidence,
not a repair of the native7B pupil or qualification of all32 teachers.

V5 stopped at28recordings:27 transport/lifecycle/frame-valid and one cancellation
failure026. Its actual arrived transcript includes the cancellation and arithmetic.
The voice answered ten tokens but created NO second delegation; therefore the backend
never received cancellation and later supplied the old bundle. Source/teacher WAVs
are not clipped. The backend contract cannot repair an absent native request.
A new generic teacher-policy experiment should prioritize dispatching pending-task
changes before answering accompanying basic arithmetic, preserving the27 valid
records, all failed parents and unchanged actual-event/full32 approval requirements.
No such new capture is yet launched. Full speech/semantic/native gates remain open.

V5 all1084files/760,659,846bytes are independently hash-verified on the persistent
volume. Current GPU readiness19/49 is confirmed live20:21UTC; numerical40 and
gradient40 passed, but0new native optimizer updates. No current teacher/API collection
remains running. See NATIVE-REASONING32-V5-TERMINAL-2026-10-23.json.
