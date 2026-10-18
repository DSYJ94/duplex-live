# Delivery target: 19 October 2026

## Scope clarification — 18 October, 19:34 UTC

Plain words: OWNER PRODUCT REQUIREMENTS clarified18October19:37UTC: chat with
useful ordinary reasoning; smooth simultaneous listening/speaking; genuine
silence instead of mandatory replies; context-appropriate stop/pause/continue
commands; and competent delegation to a stronger background reasoner. The final
foreground model must be tested for all of these together. A working text
controller is neither a prerequisite for native training nor evidence that the
native speech model can meet those requirements.

Plain words: One named native starting point: the released Context Spanning7B
checkpoint, which builds on PersonaPlex in the Moshi family and was accepted by
the owner for voice quality. This is a starting point, NOT a qualified model.
Do not restart from random weights or describe the current Qwen text run as
training this speech model. After judging the already-running bounded Qwen run,
do not launch another controller-training campaign as a substitute for native
work. Return to the native model's demonstrated failures and choose a specific
causal correction before further native updates; unchanged repeats are rejected.

Plain words: End-to-end native process: preserve the accepted pretrained voice;
build approved simultaneous-dialogue supervision for ordinary answers, silence,
pause/resume and actual asynchronous tool receipts; adapt the native model's
conversation/result-integration behavior while retaining its speech quality;
test new free-running spoken conversations including private facts, ordinary
reasoning and successful/failed/unnecessary delegation; then verify real live
latency, interruption and reproducible model loading. Silence is an intended
output, not a compulsory spoken acknowledgement. Hard reasoning stays with the
background model. If ordinary reasoning still fails, report that the chosen
base is inadequate rather than assuming more of the same updates will fix it.

Plain words: The final target remains a native duplex voice assistant whose
speech model listens while speaking, chooses whether to answer or delegate,
uses actual asynchronous results, and handles waiting, clarification and
cancellation without losing the conversation. The current implementation route
has changed: a modular fallback has become the main active training track after
the native candidates failed factual/result-use/control tests. This change of
route must be explicit; it does not make the fallback the native final product.

Plain words: Current7530021 fine-tunes786432 adapter parameters in pretrained
Qwen3.5-9B's final four layers on533 synthetic text/action labels for1599 updates.
It does not train acoustic tokens, the speech encoder/decoder, Moshi, Lychee,
PersonaPlex or Qwen Omni, and it is not pretraining from scratch. Frozen speech
recognition/synthesis and an actual external reasoning backend remain separate
dependencies. Independent tests must establish whether this control candidate is
worth retaining; it is not automatically promoted into the accepted demo.

Plain words: Finish and judge this bounded run before additional supporting
experiments. Preserve any demonstrated controller gain as a separately labelled
fallback and possible source of verified TRAIN supervision. A native follow-up
must explicitly train and evaluate the speech model itself with causally timed
user/assistant audio, actual result arrivals and the same control obligations.
Text-controller weights do not transfer directly into that model. Require new
free-running native speech holdouts, retention and latency evidence; do not infer
native success from teacher-forced loss, text-controller accuracy or smooth TTS.
Existing native failures remain visible if no viable native checkpoint emerges.

Plain words: the owner confirms on16 October that compute ends in5 days and
requires readiness1–2 days earlier. Treat19 October as the delivery target;
20 October is recovery/export buffer. Do not depend on21 October availability.
Times below are Europe/Berlin. This is a work schedule, not a claim that a
qualified model already exists.

## Time budget and decisions

| Window | Required outcome |
| --- | --- |
|16 October, remaining day|Finish the earlier actual-history continuation capture/audit. Check preserved user speech, real receipt timing, native control labels, past/span masks and reload. Stop repeating the same tiny proof once that seam passes.|
|17 October, by12:00|Freeze backbone, data version, loss and adapter recipe. Choose a bounded training run from measured memory/throughput and the useful approved corpus. Prepare the runnable fallback independently if native result use still fails.|
|17 October12:00–18 October18:00|Main adaptation and checkpoint evaluations. Use the work queue for training/batch work, interactive reservation only for permitted short engineering checks. Retain the accepted voice baseline. Stop a recipe on speech collapse, bad tool behavior or worse retention.|
|18 October18:00|Stop starting training that could consume the delivery day. Select the best passing checkpoint or explicitly record that no native checkpoint meets the task gates.|
|19 October|Cold-load and run the demo, verify complete-task behavior and latency, package custom weights/configuration/data evidence, test exported artifacts and document remaining failures.|
|20 October|Recovery and final export buffer. No new architecture or open-ended training.|

## Acceptance and honest fallback

The native candidate must produce the actual changed private value after a real
receipt, answer ordinary questions, clarify ambiguous requests, respect actual
cancellation, remain intelligible and keep listening while speaking. Check
sampled text against independent audio transcription; empty text alone is not
proof of silent PCM. Reused development scenes are not fresh qualification.
Use fresh task holdouts after the recipe is locked, with no holdout supervision.

No silent substitution of external routing, forced words, scheduled results or
a hosted gold-standard voice counts as a native-model pass. A useful fallback
may combine the accepted local voice with a separate reasoning/controller
path, but it must identify those dependencies and pass its own end-to-end
tests. Preserve the native research candidate separately.

## Export before allocation ends

Back up custom adapters, accepted voice codes/reference provenance, exact source
commits, tokenizer/codec identities, data manifests and approvals, evaluation
audio/receipts and a reproducible launch command. Verify hashes after transfer.
Public model weights can be restored from their pinned revisions; custom data
and checkpoints have first priority. The deliverable must not require an
active Isambard allocation merely to inspect or recover it.

The owner has not chosen a destination provider. Training and inference must
remain portable to AWS, GCP, Modal, Nebius or Runpod. Include optimizer moments,
completed updates, RNG states and exact recipe/data/source identity so work can
continue after expiry. No paid destination resources are provisioned yet.
See [portable environment](../portable/context/README.md). The source supports
atomic per-update state saving; exact interrupted/uninterrupted CPU tests pass.
Actual model runs7474819 and7475099 also pass fresh-process optimizer/RNG and
prediction reload. An amd64 GPU continuation remains pending. The portable
launcher supports audit/fit/resume without Slurm; no provider choice is required
to prepare or export it.

## Starting evidence on16 October

Accepted Context Spanning synthetic-reference voice remains the leading voice.
No candidate yet passes the full delegation/control task. Actual late-correction
components and head/temporal engineering proofs pass serialization and numerical
checks; temporal adapter7467496 still omits the private answer in native7467590.
Independent speech audit7467658 finds WER0.0222, so intelligibility does not
resolve the factual failure. Early captures7467981/7468102/7468611 completed;
the first passes its independent CPU audit with88 text targets and no past,
span or audio loss. Three-component temporal audit7469148 is queued. Capture
manifests stay quarantined; separate independent audit determines eligibility.
