# What we trained, what is saved, and what to test

## 24 October update — control40 pilot fit completed

The **control40 V11 adapter has completed all eight updates** on Runpod, starting
from native520's adapters and Adam state. The training process exited0 at
06:58:04 UTC. This supersedes the “not started” status in the dated23 October
snapshot below. The candidate is **unadopted**: its separate full native restore
is running, and its native behavioral evaluation has not yet run. Live process
state belongs in [HANDOFF](../HANDOFF.md), not in the historical recipe exports.

The [completed recipe](NATIVE-CONTROL40-COMPLETED-RECIPE-2026-10-24.json) is exported
from the actual final report, saved-state inspection and closed controller. It
includes the exact152-argument command, all13 adapter module names, input/audit
hashes, source SHAs, optimizer settings, parent lineage and final artifact hashes.
It is distinct from the earlier planned recipe. The full6.35MB report and saved
artifacts are preserved in the verified Mac snapshot recorded in that export.
The [input backup review](NATIVE-CONTROL40-INPUT-BACKUP-REVIEW-2026-10-24.json)
rechecks all45 derived files against the Mac, volume and actual fit bindings,
and rehashes the closed archive containing all7,797 source/approval/failed-ancestor
files. The broader stopped Mac source backup remains partial.

| Setting | Completed control40 V11 run |
| --- | --- |
| Training source | `c9efaaac24d983342f7eded46617e6450bc102ce` |
| Warm parent / Adam clocks | native520 /7510331; **520→528**, moments preserved |
| Parameters | Same **1,894,400** FP32 LoRA parameters,26 tensors, rank8/scaling2; base frozen |
| Updates / optimizer | **8**, AdamW LR2e-7, betas.9/.999, epsilon1e-8, weight decay0, norm clip1 |
| Retained objectives | Every original154 term at each update:28 histories, old25/new24 text,10 word/14 clock anchors and49 acoustic teachers; KL1/hidden10 and both64-weight anchors |
| Native streaming auxiliary | All49 once across8 updates; recorded-call forward values preserved with declared call-boundary autograd truncation |
| New control teachers | All40 once across8; all five families per update; text/acoustic epoch weights1 each |
| Execution | Native BF16 MATH, one A10080GB; no compile/CUDA graphs for fitting |
| Reported peak allocation |23,076,832,256 bytes; the recorded fit/report window, not a whole-process memory guarantee |
| Fit process elapsed |7,750.31seconds including post-fit reference generation; report timer4,385.91seconds has the narrower scope stated in the recipe |
| Saved state | Final adapters, Adam528, Python/NumPy/Torch CPU/CUDA RNG, complete report, exact input and approval bindings |

[The final fit audit](NATIVE-CONTROL40-COMPLETED-FIT-2026-10-24.json) verifies all
original objectives, all49 native teachers and all40 controls. On the fixed28
original teacher-forced history components, token accuracy changes604/782 to
601/782; the [complete denominator review](NATIVE-CONTROL40-ACTUAL28-METRICS-2026-10-24.json)
includes every component. This is training diagnostics, not conversational
accuracy. The loss evidence does not establish convergence or improvement.

The fresh-process restore must reproduce all28 original, all49 acoustic, all49
native-stream and all40 control fingerprints, plus exact Adam/RNG restoration.
Then the already waiting dispatcher runs the complete10 development cases on
baseline and candidate, followed by independent speech checks of all20 outputs.
All-case review and ordinary4/4 remain mandatory before sealed fresh measurement;
fresh measurement permission does not establish adoption. A larger training run
still requires conclusive restore, retention and native behavior evidence.

The additional32-scene reasoning/delegation collection has progressed beyond the
23 October snapshot: source audio and the complete derived teacher audio version
are recorded and speech-qualified. Its separate semantic payload approval is
pending; no semantic provider requests, physical native conversion or reasoning32
training have occurred. Control40 teaches cue silence and pause/resume, not the
whole reasoning/delegation or natural backchannel objective. No production model
or natural microturn qualification is claimed.

## 23 October snapshot — historical status

There is still no adopted production model. The Runpod A100 comparison has now
completed for both native520 and native7590052. Under the default runtime,
native520 answered2/4 ordinary core questions correctly and native7590052 answered
0/4; both failed quiet/pause and asynchronous task behaviors. The separate archived
TF32 comparison answered1/4 for each. These results replace the pending-evaluation
status in the original21 October snapshot below. Full case, speech, receipt and
waveform evidence remains in the [default review](NATIVE-PAIRED20-DEFAULT-RUNTIME-REVIEW-2026-10-23.json)
and [archived-runtime review](NATIVE-PAIRED20-ARCHIVE-RUNTIME-REVIEW-2026-10-23.json).

The current repair starts again from native520, with its Adam state. It adds40
approved control teachers across five families: no release, release with ordinary
reasoning, distractor before release, pause/resume, and replacement cue. It retains
all28 original histories, all154 original objectives, all49 existing native teacher
histories, KL1/hidden10 and the ordinary word/clock anchors. The same1,894,400 adapter
parameters are eligible for eight updates at planned effective LR2e-7. Each of the
40 new teachers is scheduled once across those eight updates; five control families
are represented at each update. The original full model weights remain frozen.

**This new training has not started.** The running pipeline first requires the
complete40 numerical audit, independent BF16 gradients and combined readiness.
Then it may train8 updates, independently restore weights/Adam/RNG/native outputs,
and run the full paired20 native and independent20 speech evaluation. All ten
candidate conversations need semantic/receipt/audio review and all four ordinary
core questions must pass before sealed fresh qualification. No successful subset,
lower loss or exact restore establishes behavioral improvement or convergence.
The [planned recipe](NATIVE-CONTROL40-PLANNED-RECIPE-2026-10-23.json) contains the
actual queued commands, clearly separated from a completed training report.
[Live evidence](NATIVE-CONTROL40-LIVE-FOLLOWUP-2026-10-23.json) and
[HANDOFF](../HANDOFF.md) carry current process and backup state.

This pilot addresses control behavior; it does not establish the broader objective
of natural microturns and reliable difficult-reasoning delegation. The additional
[32-scene reasoning/lifecycle source proposal](../evals/fixtures/native-reasoning-delegation-source-proposal-20261023.json)
is authored, not recorded, independently approved or trained. Microturn/backchannel
examples still require their own source and audio annotation protocol. The broader
[curriculum](NATIVE-INTERACTION-DELEGATION-CURRICULUM-2026-10-22.md) preserves this
scope. Explicit quiet must override every acknowledgment or filler.

## Original21 October recipe snapshot

The paragraphs and actual recipe export below describe the21 October handoff.
Their then-pending evaluation has since completed as recorded above. The exported
historical training report, weights and original failed artifacts are unchanged.

There is no adopted final production model yet. The latest research candidate is
**native7590052**, a small adapter on the released **Context Spanning DuetaSpan-7B**
full-duplex speech model, itself based on NVIDIA PersonaPlex/Moshi. The retained
comparison baseline is **native520 / job7510331**. Numerical restoration passed on
Isambard; the latest candidate's complete behavioral evaluation has not yet run.

## The cookbook in plain language

1. Start from the published speech model, its original codec/text tokenizer and
   native 80ms streaming clock. Preserve its ability to listen while speaking and
   its native context-span mechanism for asynchronous backend results.
2. Capture synthetic native histories for ordinary answers, private result use,
   clarification, cancellation, listening, lookup and acknowledgment. Validate
   actual waveform/word/timing ownership before deriving training inputs. Keep
   held-out audio separate. Repair diagnosed failures in new versions and keep
   originals, receipts and approval chains.
3. Train small low-rank adapters, keeping the original model weights frozen.
   The original28 history components and retention losses protect earlier skills.
   Old25 and new24 approved teachers add49 speech/text examples. A strict teacher,
   independent audio-quality, semantic, conversion and numerical approval chain
   controls which examples enter training.
4. Compare short pilots using native, unforced speech and asynchronous task
   behavior. Preserve failures. Earlier pilots regressed on ordinary answers, so
   neither a falling loss nor a successful checkpoint reload qualifies a model.
5. For latest job7590052, start from native520's adapters **and Adam moments/clocks**.
   Run eight small updates with the complete original154 loss components at each
   update. Add balanced teacher-text learning over all49 recorded native streaming
   histories, with the declared truncated autograd boundary at each recorded call.
6. Independently reload in a different process. Verify original28, acoustic49 and
   native-stream49 fingerprints, adapters, Adam moments/clocks, and all saved RNG.
   This passed on Isambard. Its next native10 behavioral job was rejected by quota;
   no behavioral result exists for this candidate and it is not adopted.

## Exact latest-stage settings

The machine-readable export is
[NATIVE-7590052-RECIPE-2026-10-21.json](NATIVE-7590052-RECIPE-2026-10-21.json).
It comes from the actual saved training report, not a reconstructed proposal.

| Setting | Actual value |
| --- | --- |
| Training source | `741872fe2f9e49f35d4deba9aba96caff26c51d4` |
| Published vendor source | `79df302fa2820cae10c61bb2b7f302c52edb6ed2` |
| Warm parent | native520 /7510331, existing adapters and Adam state |
| Trainable parameters | **1,894,400**, in26 FP32 adapter tensors |
| Adapter rank / scaling | rank8 / scaling2.0, equivalent rank-scaled alpha16 |
| Adapter locations | last4 temporal layers28–31: attention output and feed-forward input/output; text output head |
| Optimizer | AdamW; beta1=.9, beta2=.999, epsilon1e-8, weight decay0 |
| Effective learning rate | **2e-7**, tenfold below the retained nominal2e-6 recipe field |
| Gradient clipping | max norm1.0 |
| Latest stage updates | **8**; carried Adam parameter clocks **520→528** |
| Sampling / attention | native BF16, original TF32 setting, MATH attention; no compile/CUDA graphs |
| Original actual histories | all28 at each update, balanced across8 behavior kinds |
| Original objectives | all154 components at every update |
| Teacher histories | all25 old +24 new, balanced across4 kinds |
| Base retention | KL weight1 and hidden-state MSE weight10 |
| Ordinary retention |10 word anchors and14 clock anchors, each with retained64-weight objective |
| First response word | retained weight16; text-padding loss weight.5 |
| Assistant acoustic learning | all49 teachers,8 assistant codebooks, weight1; no user/prefix/span targets |
| Native recorded-call text | all49 once over8 updates, balanced by kind; declared truncated call-boundary autograd |
| CUDA devices for restore | expose exactly one GPU, matching the saved CUDA RNG list |

This stage deliberately began a new recipe/RNG lineage while carrying the parent's
Adam moments and clocks. It was not represented as an exact continuation of the
parent RNG trajectory. The new run saved and independently restored its complete
Python/NumPy/Torch CPU/CUDA RNG state. It did not extend rejected7530408 unchanged.

Measured GPU allocation elapsed01:36:25 on one GH200. The raw report's1871.60-second
timer includes post-fit checks/checkpoint/original28 metrics and excludes the final
49 fingerprints and independent restoration; it is not the whole allocation time.
The measured fit/report-window peak was22,071,380,480 allocated bytes. Neither number
is a Runpod throughput or capacity guarantee.

## What is the model, in terms of size?

The published model is named **7B**. The saved PersonaPlex BF16 weight file contains
**8,371,408,896 stored tensor elements**, including its depth/audio machinery; this
is a checkpoint-storage count, not a separately measured count of unique runtime
parameters. The native Context Spanning checkpoint has a similar16.74GB size.
The latest adapter adds1,894,400 trainable parameters and a7,580,728-byte file.
The original model, codec and tokenizer remain separately pinned; the7.58MB adapter
is not a standalone speech model. There is no quantized or merged release yet.

Ordinary speech comes from the native duplex model. External reasoning/tools, when
requested, remain a separate asynchronous backend; its parameters are not counted
as part of the7B speech model. The latest tested backend implementation used
`gpt-6-luna`; that is separate from the speech-model fine-tuning and does not make
the whole system an independently qualified reasoning product.

## Saved artifacts and reproducibility

Local proof directory:

`runs/cloud-export-20261016/tree/runs/context-training-proof-7590052/`

- `adapters.safetensors`: trained adapter tensors.
- `training-state.pt`: adapter values, named Adam state, complete saved RNG and exact recipe/source metadata.
- `report.json`: full recipe, data identity, loss composition, per-update exposure/metrics and checkpoint hashes.
- `reload-report.json`: independent fresh-process restoration evidence.
- `data.json`, `audio49-data.json`, voice and approved audit/reference files: exact inputs and approval lineage.

The separately pinned16.74GB Context Spanning weights,16.74GB PersonaPlex weights,
384.64MB Mimi codec and text tokenizer are already downloaded on this Mac under
`runs/cloud-export-20261016/tree/models/hf-cache/hub/`. The approved migration copies
these and all supporting files to persistent Runpod storage, then verifies hashes
there. Exact original source is recreated with Git history and vendor bundles.

Candidate adapter SHA256:
`7bbb2e9f1822cb43db50f6ead7c14d510e7c280d520cc7e1436dced6edbcfeb5`.
Training-state SHA256:
`6a9fc6cb0949d1562f654af74cb6e86a827a0214a486f869e7bd92b1ae0b63b4`.

## Testing and acceptance

Run the destination state/runtime checks first, with a new provider execution
receipt and actual Pod/device/process identities. Evaluate all10 native development
cases and independent speech, against the retained baseline. Ordinary4/4 and the
complete review remain required before extending training. The sealed20 fresh
questions require all40 paired baseline/candidate outputs and their full review.

A user audition is an experiment, distinct from those acceptance results. A native
test must actually load this adapter and its saved native layout/voice/sampler.
The old `demo/mic-demo.sh` uses scripted words from a different earlier model and
does not test this candidate. The older portable `train.sh` likewise does not
implement the current stream49/full154 stage unchanged.

Historical first approved Runpod trial (now terminated): Pod`q3wqjwols44aos`,2H100SXM,377GB host RAM,40vCPU,
volume`cyjzip53ap` at`/workspace`. Created21Oct09:15:37.923UTC; local deadline guard
terminates this trial by11:15:37.923UTC, retaining the100GB network volume. It is
$7.98/hour compute while running and$7/month storage while retained. Execution and
validation outcomes are recorded separately in the newest HANDOFF.md.


21Oct12:57UTC destination update: the initial H100 trial is terminated. Its fresh
process restored all26 adapters, Adam528 and all saved RNG exactly, with zero new
optimizer updates. Full native numerical/behavioral qualification and a usable
microphone audition remain incomplete. An approved S3-supported replacement volume
and bounded CPU staging now prepare the full backup/runtime before any GPU rental.
See the latest HANDOFF.md and NATIVE-RUNPOD-EXECUTION-2026-10-21.json for current
resource IDs, transfer ownership, spending guards and the precise next test.


21Oct14:21UTC preservation update: complete native archive independently verified
on Mac and Runpod volume49yytzyoge, all28,470files+8links. Runtime reconstructed
with Git, full pinned native source unchanged. No production model adopted and
no usable live audition yet; numerical/behavior gates remain as documented above.


21Oct14:24UTC: all task compute is stopped. Reproducible x86 package inventory:
[NATIVE-RUNPOD-RUNTIME-PACKAGES-2026-10-21.json](NATIVE-RUNPOD-RUNTIME-PACKAGES-2026-10-21.json).
Original full destination verification:
[NATIVE-RUNPOD-ARCHIVE-VERIFICATION-2026-10-21.json](NATIVE-RUNPOD-ARCHIVE-VERIFICATION-2026-10-21.json).
Both100GB volumes remain; the old one awaits unique-receipt recovery. No GPU/live
microphone path currently runs. A single A100 attempt failed at provider capacity;
no model settings or acceptance gates were changed to obtain a result.


21Oct15:29UTC: oldvolume unique receipts recovered, all25 files85.6MB independently
verified onMac and newvolume; oldvolume retired. Ongoing storage is single100GB
49yytzyoge/$7month. Original H100 restore receipt now committed in
[NATIVE-RUNPOD-INDEPENDENT-STATE-RESTORE-2026-10-21.json](NATIVE-RUNPOD-INDEPENDENT-STATE-RESTORE-2026-10-21.json).
A bounded1A100 voice trial is booting; see newest handoff for actual readiness.


21Oct16:57UTC: actual native audition is live onMac via http://127.0.0.1:8765,
automatic stop17:28UTC/18:28London. Full strict model/adapter/voice load and302frame
synthetic streaming PASS after installed CUDA header/assembler path repairs,
with original BF16/native clock/sampling preserved. Source909c547; no newupdates.
The seenTRAIN008 response is WRONG (invented emptybox), so this is experimental,
not qualification/adoption. External tools disconnected, no owner recordings saved.
Exact original launch and sample result:
[NATIVE-RUNPOD-VOICE-LAUNCH-2026-10-21.json](NATIVE-RUNPOD-VOICE-LAUNCH-2026-10-21.json),
[NATIVE-RUNPOD-VOICE-SMOKE-2026-10-21.json](NATIVE-RUNPOD-VOICE-SMOKE-2026-10-21.json).


21Oct17:20UTC: the owner reported broken silence-until-keyword control. Complete
synthetic diagnostics reproduce0/9 quiet-text passes, despite9/9 working streams;
two controlled prompt changes did not solve it. This is instruction-following/
state-tracking failure and blocks adoption/unchanged extension. See the exact
[diagnostic review](NATIVE-KEYWORD-CONTROL-DIAGNOSTIC-2026-10-21.json) and
[new-stage proposal](NATIVE-KEYWORD-CONTROL-NEXT-2026-10-21.md). No new training or
masking workaround has been performed. All failures and51artifact hashes preserved.


Adapter capacity clarification —22October2026: the1,894,400 count was independently
recomputed from all26 tensor shapes in the actual native7590052 safetensors file;
see `NATIVE-ADAPTER-CAPACITY-2026-10-22.json`. It is an exact inventory, not evidence
that rank8 or final4-layer coverage is optimal. There are13 adapted matrices:
attention output and gated feed-forward input/output in layers28–31, plus the text
head. Each matrix adds rank×(input+output) trainable values; total is
4×(65,536+212,992+122,880)+288,768=1,894,400. Base matrices, QKV parameters and
acoustic depth/codec weights are frozen; temporal changes can still change the
conditioning reaching speech generation. This selection is a constrained engineering
hypothesis; no matched layer/rank capacity comparison has established sufficiency.
The entire native model executes during training/inference. A successful adapter
pilot does not schedule full-weight fine-tuning automatically: retain/extend this
method if heldout losses and native behavior improve, or test broader adaptation
if evidence indicates a capacity limit. Partial/full-weight training would require
a separately specified data, memory, optimizer and compute plan. The intended
product capabilities, not the adapter count, define completion.
