# What we trained, what is saved, and what to test


## 25 October — saved8 recovered exactly; full behavioral evaluation rejected

The exact saved eight-update reasoning32 candidate has now passed zero-update
finalization and a distinct fresh-process full Adam/RNG/logit restore. No optimizer
updates were repeated. Original fit exit−15 remains recorded. The independently
recomputed old28 kind-balanced weighted loss worsened from0.5675211192 to0.5720761260
(+0.8026%); the pre-update step8 trace was not a final loss or convergence result.
See [complete recovery](GCP-NATIVE-SAVED8-RECOVERY-COMPLETE-2026-10-25.json).

Both models completed all ten native development cases, followed by independent
speech audits of all twenty original output WAVs. Basic ordinary correctness is
2/4 for native520 and1/4 for saved528. The candidate gets photosynthesis correct,
but gives wrong ice/sound explanations and answers the metal question with water
freezing. Independent speech confirms those errors. It fails pause/resume, clean
result use and actual cancellation; repetition remains. Speech agreement/bounds
passes8/10 baseline and7/10 candidate at the unchanged WER≤0.08/bounds rule. These
counts are not full-task qualification. All10 cases/both roles are reviewed in
[the complete rejection](GCP-NATIVE-SAVED8-COMPLETE20-REVIEW-2026-10-25.json).

No fresh payload was opened, candidate adopted or training extension run. Every
paired user PCM matches, both native cold/intervening/reused parity checks pass,
and all user samples were processed without forced words/requests. Steady compute
p95 is52–59ms per80ms frame, but approximately0.5s initial backlog and one113–126ms
playout gap per run remain; this is not microphone/product acceptance.

The full evaluation backup is324 regular files/717,377,986 bytes plus52 driver-link
records, all remote hashes/link targets verified on the Mac. The exact A100 was
stopped early; provider confirms TERMINATED at07:06UTC, same250GB disk READY and
non-auto-deleted. Critical checkpoints/optimizer/RNG/data/receipts remain preserved.
No further GPU is running for this work. Current grant authorizes zero new updates.
The next repair must change a diagnosed cause and retain all histories/losses,
independent restore, ordinary4 and full native/fresh gates. Do not extend this
rejected stage unchanged or repeat earlier persona-only experiments as a cure.


## 24 October — eight updates preserved; post-fit checks timed out

All eight reasoning32 optimizer updates were saved. The guarded fit then ended
at 17:10 UTC during post-fit fingerprint collection: all49 retained native
fingerprints and31/40 control fingerprints were written. The fit report,
independent full restore and native behavioral evaluation did not complete.
The time allowance was too short for those post-fit checks; this outcome does
not supply a behavioral verdict for the new candidate.

The adapter, full optimizer/RNG checkpoint, exact eight update receipts and
all retained154/native49/control40/reasoning32 exposure are hash-verified on the
Mac. See [the closed result](NATIVE-REASONING32-POSTFIT-TIMEOUT-2026-10-24.json).
The dedicated Runpod machine is stopped, with the100GB network volume retained.
No additional GPU or provider transfer has been started.

The [raw retained training trace](NATIVE-REASONING32-RETAINED-TRAINING-TRACE-2026-10-24.json)
contains pre-update values, not a final post-update evaluation. The original28
weighted loss is0.56752112 before update1 and0.56746160 before update8; the sum
of retained154 weighted group losses rises from2.55257399 to2.68811505 over the
same observations. This mixed trace does not prove convergence or better voice
behavior. The completed matched initial/final report remains unavailable.

The next step is a separately verified zero-optimizer recovery and full restore,
then complete native/speech evaluation. The original timeout remains preserved;
no repeated optimization, subset acceptance, extension or adoption is authorized
by the saved-stage audit. GCP continuation has a concrete proposal and awaits the
new usage-and-transfer decision.


## 24 October — reading the reasoning32 fit results

The current status is in [the fit record](NATIVE-REASONING32-FIT-LAUNCH-2026-10-24.json)
and the newest `HANDOFF.md` entry. The dated entries below preserve earlier observations.

To measure the effect of this corrective stage, compare `initial_stage_metrics`
with `final_metrics` in the completed training report. The initial stage metrics
are recorded after the native520 adapter and parent Adam state are loaded, before
these eight updates. `baseline_metrics` is recorded before that warm start; it is
not the native520 comparator for the effect of this stage. The implementation is
`evals/context_actual_train.py` at the fixed model source `da389a7`.

Compare the same ordered 28 actual-history components and report the chosen NLL
aggregation explicitly. Keep `nll`, `optimization_nll`, `non_pad_nll` and token
`top1` separate: they use different target weightings or subsets. Per-update
reasoning losses use different four-teacher batches and do not form a matched
before/after evaluation. Raw retained-batch weighted losses also have a different
aggregation from a simple mean of final per-component NLL values.

Lower teacher-forced loss is evidence about those training targets. It does not
prove free-running silence, pause/resume, correct reasoning or real cancellation.
The independent full training-state restore, complete native20/speech20 review
and gated fresh evaluation remain required before extension or adoption.


## 24 October — first reasoning32 optimizer update saved and backed up

The [first real optimizer update](NATIVE-REASONING32-FIRST-UPDATE-2026-10-24.json)
is complete: stage step1/8 was observed at15:16:44UTC. The fit remains running.
Its pre-clip gradient norm is16.85215950012207, retention KL0.06867368519306183
and hidden MSE0.011277739889919758. These are pre-update training metrics;
there is no matched after-update loss or behavioral improvement claim yet.

The22,970,059-byte training-state checkpoint is preserved in the verified Mac
copybackV3 snapshot20261024T151246.091547Z with SHA256
5ff1a415dafe8bd450418bec3d78d1554d425c9b20bc3cbb765804fa5ef70716.
The Mac file hash matches its verified remote snapshot inventory. Full independent
training-state restoration remains pending after all eight planned updates.
The baseline, frozen weights, original objectives and all quality gates remain.

## 24 October — complete readiness passed; reasoning32 fit process launched

The [complete joint readiness audit](NATIVE-REASONING32-JOINT-READINESS-COMPLETE-2026-10-24.json)
passed all eight partitions and exited0 at14:48:23UTC. It combines the unchanged
original154, native49, control40 and new32 objectives at native520. Model parameters
remained unchanged; peak allocation was23,088,837,120bytes. This proves execution
readiness, not learning or behavior.

The original controller then refused its first training launch solely because
its conservative start window had closed. The staged followup reused the complete
approvals and [launched the same eight-update fit](NATIVE-REASONING32-FIT-LAUNCH-2026-10-24.json)
at14:53:47UTC. Controller28726/fit child28807 use unchanged model sourceDA389A7;
controller source0d75986 changes orchestration only. The model command differs
only in output directory. All hyperparameters in the planned recipe remain.
At15:03:47UTC the fit child was live and its actual command matched; no optimizer
step had yet been logged. Do not report eight completed updates or convergence.

A new local evaluation waiter20887 follows exact controller PID/birth and requires
complete training, independent restore and strict proof before full paired native20
and independent speech20. Its latest launch is17:40UTC, retaining the existing
minimum40-minute window and unchanged18:20UTC evaluation deadline. All cases and
thresholds remain required; sealed fresh inputs stay closed. Backup8236 covers
V17 and V18. The Pod hard stop remains18:39:58.8UTC; compute cap is unchanged.

## 24 October 13:08 UTC — numerical and gradient approval complete; fit still planned

The new reasoning32 corpus passed complete32 numerical and gradient approval.
[Numerical evidence](NATIVE-REASONING32-NUMERICAL-COMPLETE-2026-10-24.json)
and [gradient evidence](NATIVE-REASONING32-GRADIENT-COMPLETE-2026-10-24.json)
record actual process exits, unchanged model parameters and final report hashes.
These checks establish that the objective executes correctly; they do not show
learning, convergence or behavioral improvement.

The exact queued command and hyperparameters are saved in the
[planned recipe](NATIVE-REASONING32-PLANNED-RECIPE-2026-10-24.json).
This pilot has **zero optimizer updates** at this observation. Controller8514,
source da389a7031a5504495397d1575952194ffa209f1, is running combined readiness
on Pod8fsk2kouknoiac. Its child15566 is live; the original154 complete-gradient
preflight passed and29/49 retained native histories have logged exact forward
hash parity. Complete49 and all eight joint partitions are still required.

The later [retained-readiness substage report](NATIVE-REASONING32-RETAINED-READINESS-2026-10-24.json)
records all49 PASS, in addition to original154 PASS. Both final reports have exact
remote/Mac hash equality and the pure complete49 gate was rechecked locally.
The eight joint partitions remain in progress; this does not change the planned
status of training or imply behavioral improvement.

The queued fit starts from native520, trains1,894,400 adapter parameters in26
FP32 tensors across13 modules (rank8, scaling2), and keeps the7,689,623,552
base parameters, Mimi and depth transformer frozen. It plans eight AdamW
updates, effective learning rate2e-7 (2e-6 times0.1), betas0.9/0.999,
epsilon1e-8, weight decay0 and gradient norm clip1. Expected Adam step528
is conditional on all eight updates actually completing.

Each update retains the original154 objectives, original28 histories and
approved old25/new24 histories. KL1, hidden retention10, word-clock anchor64
and first-word weight16 remain. Native49 and control40 each cover the complete
set per eight updates. The added32 scenes cover direct basic reasoning,
delegation, cancellation and replacement: four scenes per update, one from each
family, all32 once per eight updates. Selected teacher weight is0.25; text and
acoustic epoch mean weights are1 each. The pinned upstream assembler's exact24
pre-span acoustic exclusions remain explicit; source audio and words are unchanged.

After fit, a separate process must restore full training state and verify it.
Complete paired native development and independent speech evaluation follow;
strict behavioral and fresh qualification remain necessary before extension or
adoption. The earlier rejected control40 candidate remains unadopted. No broad
reasoning, natural-backchannel or production-quality claim follows from readiness.

## 24 October09:20 update — model rejected; new32 data ready through codec

The completed8-update control40 model remains unadopted:2/4 ordinary cores and
failed quiet/pause/delegation behavior. The preliminary adapter-disabled base score3/4 was too generous: strict complete
explanations yield1 complete,2 partial and1 incorrect. The
[explicit scoring correction](NATIVE-CONTROL40-BASE4-SCORING-CORRECTION-2026-10-24.json)
preserves every original output and frozen expectation. Ice already fails in the
base; V11 loses the older adapter's complete heat-from-skin explanation.
Batch size, a particular loss coefficient and a longer run are not isolated fixes.

The additional32 reasoning/delegation episodes now have complete speech, semantic,
conversion and independently verified native codec approval. Two diagnosed ASR
clock errors were repaired in a new derived version using the same unchanged
recordings;30 exact original passing reviews were retained and two new reviews
passed. All40 actual request markers and24 delivered contexts remain.
[Complete evidence and hashes](NATIVE-REASONING32-COMPLETE-CONVERSION-CODEC-2026-10-24.json).
These32 have received **zero native training updates**. Engine-input, numerical,
gradient and combined retained-objective approvals remain before a new fit.
Natural backchannels and production performance remain unqualified.

The earlier24 October status below records the progression before this update.

## 24 October update — control40 pilot fit completed

The **control40 V11 adapter has completed all eight updates** on Runpod, starting
from native520's adapters and Adam state. The training process exited0 at
06:58:04 UTC. This supersedes the “not started” status in the dated23 October
snapshot below. The candidate is **unadopted**. Its [complete fresh-process native restore and
strict proof](NATIVE-CONTROL40-FULL-RESTORE-PASS-2026-10-24.json) passed at07:51 UTC.
The complete paired native/independent speech evaluation has finished. Both models
answer2/4 ordinary cores correctly. The candidate violates cue silence and pause,
fails both changed-value deliveries and lacks a real cancellation action. The
[complete20 review](NATIVE-CONTROL40-COMPLETE20-BEHAVIOR-REVIEW-2026-10-24.json)
rejects fresh measurement and training extension. Live process
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

The fresh-process restore reproduced all28 original, all49 acoustic, all49
native-stream and all40 control fingerprints, plus exact named Adam state and all
four RNG families. All seven pipeline stages actually exited0. The dispatcher
then launched the complete10 development cases on baseline and candidate, followed
by independent speech checks of all20 outputs.
All-case review and ordinary4/4 remain mandatory before sealed fresh measurement;
fresh measurement permission does not establish adoption. A larger training run
still requires conclusive restore, retention and native behavior evidence.

The additional32-scene reasoning/delegation collection has progressed beyond the
23 October snapshot: source audio and the complete derived teacher audio version
are recorded and speech-qualified. The user authorized the exact separate
semantic payload; all32 actual independent reviews completed, with30 passing and
two timing failures (029/030). No physical conversion or reasoning32 training has
occurred. The entire collection remains unapproved; failed originals and all32
responses are preserved. Control40 teaches cue silence and pause/resume, not the
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
