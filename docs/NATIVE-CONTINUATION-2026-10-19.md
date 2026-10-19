# Local native continuation — 19 October 2026

Local is the active executor; cloud must not submit until explicit owner takeover.
The 21 October allocation deadline remains. Native/whole-system verdict FAIL.

## Access and preservation

Official local Clifton sign-in renewed the local certificate to 22:51:23 Berlin
19 October. Actual bounded SSH succeeded at08:51:48UTC; account squeue empty.
7531496 COMPLETED; 7531499 FAILED(1:0); 7531501/02/03 CANCELLED. No duplicate jobs.
Source baseline4ad4367, ownership handoff156b9cf pushed to main.

Completed the stopped four-directory source backup and compared all scoped
remote/local files by size and SHA256: 1,450 files / 661,640,011 bytes, all match.
Scope also includes native7510331 checkpoint/optimizer/RNG state, rejected
7530408 checkpoint/state, 28 old actual-history components, scope7508779,
streamed native approval7508795 and numerical7509062 plus launch/audit receipts.
This is a scoped verified backup, not a claim that every cluster asset is exported.
Ignored Mac evidence: runs/local-continuation-20261019/{remote-critical-inventory.json,
backup-verification.json,source-diagnosis.json,failed-source-waveforms.png}.
Portable artifacts: runs/cloud-export-20261016/tree/. Credentials, owner sessions
and quarantined core are excluded. Existing full model/runtime export remains
separate; approved teacher source/cache dependency coverage is being refreshed.

## Reproduced source failures and bounded repair

Command: PYTHONDONTWRITEBYTECODE=1 .venv-api/bin/python -m
 evals.native_v2_source_repair_v2 --source
 runs/cloud-export-20261016/tree/runs/native-v2-source-repair-7531496
Expected exit1 reproduces the actual failed sources, with exact audited hashes.

TRAIN015 index3 Go. is3.05seconds and contains two distinct acoustic bursts.
Independent audit measures Go. Go. at0.10–0.42 and1.48–1.80seconds; raw RMS in
those intervals is0.123345 and0.146301. This is actual repeated acoustic content,
not just an unsupported ASR repetition hypothesis.

TRAIN017 index0 What is the parcel label? is7.4seconds with50 samples >=0.999
at3.541833–3.741125seconds. The entire raw first utterance equals both the user
stem and mixed waveform at the declared800ms offset. Clipping originates in
raw generated audio; it is not introduced by mixing/overlap. Repaired index2 is
clean. Additional acoustic bursts after the independently aligned question remain
visible; rerender the diagnosed raw source rather than lowering gain/cropping.

Repair v2 renders only TRAIN015/index3 and TRAIN017/index0, exact sealed words
and same voices. General instruction asks for one reading and silence after the
last word. Each gets one successful generation, never a local ASR reroll loop.
Original failed WAV/receipt are copied under repair-originals/; originals remain
in prior directory. Every other utterance and held-out audio/receipt stays hashed
unchanged. Rebuild only the two TRAIN scenes and their manifest durations.
No thresholds change. All24 TRAIN scenes must pass independent full re-audit.

22 focused tests pass, including original/held-out preservation, fail-closed
held-out corruption and destination overwrite rejection; Slurm shell syntax and
diff whitespace checks pass. No model qualification follows from these tests.
Continue teacher/independent quality/semantic/conversion/native numerical chain
only after the complete source gate. Retain old histories, four-reference KL and
hidden retention and25 approved native teacher histories. Do not extend rejected
7530408 unchanged. Independent state restore and full native behavioral evaluation
remain mandatory before extension or adoption.

## Source gate passed; teacher collection active

Repair7541873 COMPLETED(13s); full source audit7541875 COMPLETED(88s).
All24 TRAIN scenes pass;36/40 total scenes pass. Four held-out failures remain
unchanged/excluded and do not constitute a full qualification set. Teacher7541876
is collecting all24; independent quality7541878 and semantic/conversion7541879
remain afterok dependencies. No optimizer submitted. Source1519b71 remains the
immutable chain implementation; HANDOFF ownership/job ledger486a203 is pushed.

Dependency backup169files/280,281,639bytes also matches all remote/local hashes.
The native base asset backup is rehashed: context_spanning_7b.pt16,743,028,866bytes
SHA25632a4dec8ea8feb28d20ce485f0e49a09b118413aaaebdd182379db41cd441fcc;
README and released voice hashes also match the pinned manifest.

Prepared immutable wrappers native-v2-encode.slurm and
native-v2-native-approval.slurm support a separate complete24-teacher codec,
scope, streamed-input, independent CPU and numerical approval chain. These are
not yet executed. They leave the old25-teacher loaders/guards untouched. Every
future submission must use committed/pushed source and an exclusive receipt.
Capture uses a complete0-23 array with maximum4concurrent GPU tasks; CPU approval
must consume all24, never a subset. Original FP32 numerical failure must be kept
before any separately requested higher precision reference audit. Encoding is
for approved teacher audio only; it does not authorize an optimizer update.

## Prepared native eight-update path (not executed)

The next bounded native candidate preserves the7510331 recipe exactly: all28
actual-history components, first-response-word weight16, four-reference KL1 and
hidden MSE10, all25 independently approved native teachers with weight1 and
balanced spoken/clock text loss, native520 warm-start weights and accepted voice.
It starts a NEW optimizer, rather than claiming520+8 resumed updates.

A separately approved COMPLETE24-teacher corpus adds one weight1 balanced native
assistant-text/control/silence loss, on actual streamed-user conditioning. This
is a controlled new-data text-supervision experiment in the native model, not a
Qwen controller and not the rejected standalone joint audio pilot. There is no
new acoustic prediction loss in this candidate. Source words/assistant acoustics
remain independently approved conditioning; base acoustic parameters stay frozen.
This choice retains the successful pretrained voice while isolating the new
ordinary/quiet/pause/delegation supervision with every old retention term intact.

New flags require the entire cache/input/scope/numerical chain, old retention,
all25 teachers and an eight-update warm start. Longer updates or resume fail
closed until a separately reviewed behavioral extension is implemented. Existing
25-teacher guards remain unchanged. Additional policy and packaged approval live
under additional-native-v2/ and enter exact optimizer/RNG state metadata, so a
restore cannot silently change data/objective. Wrapper native-v2-train.slurm
runs a separate full state/logit restore process after the eight updates.
26 focused local tests pass; actual GPU execution and behavior are still pending.

## Diagnosed independent ASR onset fault

Read-only waveform comparison finds six first-batch word onsets before ANY
nonzero recorded PCM by more than200ms (003/005/006/007/008/015). In008, If starts
at0.08s while first nonzero PCM is0.748667s and detectable voice is5.76s. In015,
Put starts0.06s while first nonzero PCM is0.768042s and detectable voice6.26s.
Thus the original5/24 diagnostic admits two recordings with unusable onsets;
that number never constituted conversion/native training approval.

The separate opt-in ASR repair applies the EXISTING20ms/0.001activity floor and
250ms margin to leading quiet, then adds the exact window offset back to every
word/segment interval. No original WAV, internal gap, receipt clock, word,
quality threshold or prior report is rewritten. Original audit stays failed.
It also rejects word onsets preceding any recorded nonzero PCM beyond the200ms
bar. Regression tests prove voiced samples/internal gaps are retained and a
known2.0s onset maps back exactly. The original physical-onset check was run and
failed on those six real recordings; corrected actual re-audit remains pending.
Both original quality and semantic approvals must be repeated for any new labels.
Old25 approved native histories/targets remain unchanged.

## Reference-conditioned TRAIN-label collection (prepared)

Teacher-v3 original audit7542690 admits9/24, so general full-sentence wording
alone does not make a complete corpus. Some raw model transcripts still omit
required concepts (008 In the blue box.;010 downward instead of down). A timing
repair cannot fix missing transcript words while exact speech agreement remains
mandatory. No original result is silently accepted under a different gate.

Explicit teacher protocol native-dialogue-v4 supplies the independently AUTHORED
TRAIN ordinary QA references (the six source QA pairs plus the authored timer
answer), never captured native/evaluation responses or held-out answers. It also
supplies TRAIN project-name vocabulary for listening, with an explicit ban on
inferring missing identity. NO private value is supplied. Actual private reads,
cancellation receipts, waiting/quiet behavior, all24 audio/fact checks and actual
Astra semantic review remain mandatory. Tests prove no held-out QA or synthetic
private phrase occurs in the reference instructions. Teacher environment and
manifest carry an explicit reference-conditioning marker; semantic review sees
that provenance. Old protocols are preserved unchanged.

This constructs supervised labels from independently authored source references;
it does NOT establish unaided GPT-Live reasoning or native model intelligence.
The native student sees its original native inputs and approved target histories,
not this teacher reference prompt. Free-running ordinary/private/clarification/
cancellation/quiet/pause-resume tests remain mandatory before extension/adoption.
New reference-conditioned collection has not yet been submitted at this entry.

## Material decision: four old approved histories have impossible onsets

Read-only SHA-bound comparison of all25 old approved native histories finds four
with first words preceding ANY nonzero recorded PCM by>200ms, and preceding
20ms RMS>0.003 activity by6.06–6.68seconds:

- teacher-cf-recovery-train-001-0: Okay,0.10s, first PCM0.832708s, voice6.32s.
- teacher-cf-recovery-train-002-0: Sure,0.06s, first PCM0.830125s, voice6.74s.
- teacher-cf-recovery-train-007-0: Okay,0.10s, first PCM0.790333s, voice6.16s.
- teacher-recovery-train-004: Sure0.10s, first PCM0.800667s, voice6.26s.

All25 original WAV hashes match their approved conversion receipts. No original
word, waveform, receipt, target/cache or scope was changed. The existing native
520 baseline remains preserved for comparison; this finding is not proof that
these four timing errors caused every prior behavioral failure.

Evidence: docs/NATIVE-OLD25-ONSET-AUDIT-2026-10-19.json. Proposed repair retains
ALL25 histories and ALL28 actual-history components plus unchanged KL/hidden
retention. Only the four defective derived timing/cache/input histories receive
new versioned measurements, followed by full independent25-teacher approval and
native numerical preflight. Originals remain archived and hash-bound. Every new
24-teacher scope/approval remains separate. No optimizer should consume the old
invalid target timing while this decision is pending. Independent source/data
qualification and preservation continue; no native checkpoint is adopted.


## Owner authorized full25 timing repair — 19 October 2026 10:00 UTC

The owner approved "Repair and independently reapprove all 25 (recommended)."
The four diagnosed derived histories can receive new timing/cache/input versions;
all25 exact original words/audio/receipts and all28 actual student histories remain
mandatory, with unchanged KL1 and hiddenMSE10. No optimizer before full25 native
reapproval. New schema dl-native-old25-timing-repair-v1 explicitly enforces this
inventory; legacy loaders and rejected joint pilot remain unchanged.

Fresh CPU ASR measures only the four diagnosed unchanged recordings with the
already controlled leading-window/original-clock procedure. Existing21 measured
intervals must remain identical. Full25 current speech/fact/tool/clock checks are
rerun; conversion copies all25 originals and changes only four derived sidecar/
timeline intervals. It cannot change a word, waveform, fold or worker receipt.
Tests exercise exact label preservation and reject added/missing/reordered words.
Eight focused local tests pass in .venv-api; system Python lacks NumPy.

Final reference-conditioned teacher-v4 audit7543297 fails at13/24; semantic7543298
was cancelled by its failed dependency. No optimizer. Remaining11 failures include
an actual private phrase omission, word disagreements, paused-step omission and
one delivered cancelled result. This requires diagnosed label-collection repair,
not threshold changes or a passing subset. Final report backup2files/233652bytes
is fully SHA256 verified. V3 corrected audit7543296 admitted11/24 only.


## CPU native-cache repair and full25 GPU approval prepared

The pinned native text Interleaver can run independently of Mimi encoding. New
native_old25_timing_cache.py first reproduces each faulty ORIGINAL text lane
exactly with that pinned implementation, then reinterleaves its freshly measured
clock. Every original 16 acoustic lanes, frozen reference tensor, actual arrival,
marker and frame count must remain equal. The other21 PT files are copied byte
for byte. No acoustic re-encoding or new ARC output is needed. Complete25 lineage
is validated on every cache load; extra/omitted cases fail closed.

Separate native-old25 wrappers require the full25 scope, a new0-24 streamed-user
capture array (maximum4 GPU tasks), independent CPU input approval and native
numerical preflight. Old baseline artifacts and rejected joint-pilot guards remain
immutable. The cache job includes regression checks rejecting changed acoustic
codes, references, tool clock, markers, shape or dtype. No optimizer submission
is included in this approval chain. All source deployment uses pushed Git SHAs.


## Repaired25 cache passes; native scope wrapper import defect

7543730 passes all25 acoustic/reference/receipt/frame invariants; four text clocks
changed, zero acoustic codes changed. Native scope7543731 fails at its initial
cache validator import: pinned finetune Interleaver imports moshi.conditioners,
but this wrapper exposed only Context Spanning/env dependencies. The cache job
already had the correct pinned Moshi path. Add the same path to scope/input/math
and future training wrappers; reuse the verified cache, cancel no running GPU
work, and create a fresh exclusive approval receipt after terminal-state checks.
No native optimizer ran. GPU dependency jobs3732/3733/3734 cannot pass the failed
scope gate. This is execution plumbing, not a changed data or numerical threshold.

The bounded native8-update preparation now accepts the repaired25 cache separately
from the original three caches used for all28 actual histories and four KL/hidden
references. Strict policy comparison allows only four diagnosed timing target
hashes and new cache/approval identities: all25 words/audio/masks/frame counts,
all21 undiagnosed target hashes, objective/weights/coverage remain identical.
Missing repaired cache, dropped teacher, audio/frame/undignosed-target change,
zeroed retention or extended updates fail before gradients. Eleven focused local
tests pass. Actual native execution and independent restore/behavior remain required.

Teacher repair7543779 rerenders only the11 failed v4 TRAIN recordings after CPU
7543777 passes58tests; full24 quality7543781 and semantic7543783 remain afterok.
Sourcebddd53b and receipt native-v2-teacher-v4-repair-local-20261019.json identify
this exact attempt. Original13 passing recordings and all source/fold/heldout
hashes are retained. No passing subset qualifies either corpus or native model.


## Corrected native25 scope PASS; capture array executing

Source d4de305 CPU7543887 passes59tests. Scope7543889 independently verifies
all25 original words/audio/masks/frame counts and all21 undiagnosed target hashes
unchanged. Full0-24%4 capture7543890 is executing native user encoder/agent decoder
on allocated Isambard GPUs. CPU input approval7543891 and FP32 math7543892 remain
afterok. This is supervised-input approval, not free-running task qualification.

Full24 teacher retry quality7543781 fails; dependent semantic7543783 cannot run.
Do not accept a subset. Scene015 actually starts Put the during the ongoing
9.35s user instruction, then aborts when After I say go arrives, acknowledges
Noted and later says Close the cupboard. Source user ends10.15s, pause starts
13.44s, so a3.29s response window exists. This is premature/abandoned first-step
behavior, not proof that the source pause clock is impossible. Preserve original
source timing and all raw recordings; diagnose remaining audit failures first.

Prepared native-v2-development fixture/wrapper declares ten post-restore native
cases before any new optimizer results: four old ordinary questions, two actual
quiet/pause TRAIN scenes, paired private values, clarification and cancellation.
Native outputs are unforced; backend reads/cancellation execute from native RET
and actual arrived-audio ASR. All ten plus independent audio/receipt/clock review
are mandatory before any extension. Reused inputs are development evidence, not
fresh full-task qualification. No behavior or optimizer jobs submitted yet.


## Runtime preservation verified; remaining five TRAIN teacher defects

Runtime retry completes10:22:29UTC:17104regular files/9,072,299,252bytes all match
cluster sizes/SHA256;8symlink targets match separately. Final SIF8,551,931,904bytes
SHA256c00bc99693e4908ea75d17df9c379b93ae9a1678a65d875565ad5d52f76fa27b.
Only this task's obsolete interrupted2.2GB temp removed after final verification.
Known critical/state/data/base/runtime preservation is verified; this does not
claim that every unscoped cluster file or historical partial Mac export is complete.

Teacher retry audit7543781 is19/24. Remaining failures001/014/015/019/022 are exact
word disagreements except015's actual premature/abandoned first step. Versioned
clarity retry binds the exact new parent/check SHA256s, rerenders ONLY these5,
preserves19 passing raw records, all source words/audio/folds and earlier archives.
Per-record native-dialogue-v5 is explicit in the actual environment, manifest and
semantic provenance. It enunciates words/articles and waits for the complete
multi-clause first-step instruction rather than emitting a fragment and Noted.
It uses only already authored TRAIN ordinary/step references and actual private
receipts; no private value or held-out answer is included. All24 strict gates are
rerun.31 focused local tests pass, including changed-passing/undeclared-protocol/
source-edit/subset rejection and heldout/private-reference exclusion.


## Retained offline controls and explicit native streamed-user capture

7543890_0 and_1 fail the historical offline-input equality check, writing complete
raw native traces;_2/_3 also finish. The independent native approval already knows
how to rebuild prepared inputs from actual streamed user codes, preserving every
assistant word/acoustic input/prefix/mask and checking all17 actual input channels.
Do not assume this accounts for the observed failures until that independent CPU
check runs on the four traces. Other hypotheses are assistant delay/prefix changes
or mismatched repaired-cache assembly; the same exact tensor comparison separates
them. Cancelled4–24 and dependent3891/3892 before more GPU controls execute; receipt
native-old25-capture-interruption-local-20261019.json records actual prior states.

New explicit capture-only mode retains the original failed offline comparison and
requires exact prepared-versus-actual equality using the real user encoder output.
Legacy mismatches outside user streams9–16, any ownership or frame difference, or
any streamed-input difference still fail. There is no Engine/decoder/target change.
Four old traces remain immutable controls and can be reused only after independent
validation; remaining21 are captured separately, then all25 reapproved together.
Independent CPU full approval now fails closed on any subset before emitting its
approval schema. Explicit controls use a different diagnostic schema rejected by
the native training loader. This strengthens the all25/all24 gate, not a numerical
or quality threshold change. Regression coverage includes assistant contamination,
missing frames, wrong native user code, lost ownership and diagnostic training use.


## Clarity23/24; independent control wrapper correction

Full quality7544226 measures23/24 passes. The first-step/quiet/resume015 now passes
without source edits, as do001/014/019. Only022 fails word agreement; semantic4227
is cancelled. Keep all23 passing recordings, inspect022's exact raw/model/ASR
words and rerender only that recording in a new version if diagnosed. No subset
conversion or optimizer approval follows.

CPU7544406 passes67tests. Independent control7544409 never reached validation:
the new shell phase branch used [ ... || ... ], which bash -n cannot detect as a
runtime test error. Reproduced the actual source condition failure, corrected it
to [[ ... || ... ]], and executed the exact approve/control/math branch condition
successfully. Original four traces/failed controls stay immutable. Retry only the
CPU control from a new pushed source/receipt, with no GPU or optimizer submission
until all four actual streamed-input comparisons pass independently.


## Same-word orthography caused the last23/24 rejection

Actual022 model and independent audio word measurements differ only in US/UK
Canceling/Cancelling and canceled/cancelled. Raw normalized-token WER is0.095238;
existing spoken_words already equates canceled/cancelled, giving0.047619 but still
rejecting the missing inflection spelling pair. With canceling/cancelling as the
same spoken word, the finite canonical score is0.0. No missing/added word or
semantic substitution is involved. Do not rerender or edit this clean recording.

Explicit opt-in finite-orthographic policy records both raw and canonical scores,
checks the exact declared mapping, and retains all raw model/ASR text and measured
word timestamps. The0.08 bar and exact spoken-word, fact, receipt, physical-clock,
quiet and cancellation gates stay unchanged. Legacy raw scores remain default.
No synonym, dropped article, repeat or content substitution can be normalized.
Regression examples Go→Go.Go, bus→boss, missing A, take in→absorb and cancelled→
complete all still fail. Forged raw/canonical scores or a boss→bus map are rejected.

On actual saved24 records, only022 changes verdict and every measured word/time,
raw record, reference arrival and delegation is unchanged;34 focused tests pass.
A new independent full24 same-WAV CPU re-audit must reproduce original raw WAV/
ASR/model/word timing payloads exactly and retain all23 existing passes before
semantic/conversion. Original failed audit7544226 remains immutable. New approvals
still require all24 semantics/conversion/native numerical gates, never a subset.

CPU input control retry7544729 uses pushed source1230c83. The prior afterok request
was refused because completed test7544406 aged out of Slurm's live dependency
controller; verified archived COMPLETED0:0 and identical tested Python bodies,
then submitted once without a stale dependency under the same exclusive receipt.
No duplicate job or new GPU submission. Four raw native controls remain preserved.


## Fresh full-task qualification plan declared before candidate outputs

New native-fulltask-qualification-20261019.json declares20 authored user-only
cases:8 ordinary reasoning/facts,4 actual private reads including identical-input
counterfactual worlds,2 clarification,2 actual cancel-plus-new-answer,2 explicit
quiet cues and2 paused/resumed steps.38 user clips/1951characters stay within the
existing generation budget. No audio generation/evaluation job is submitted yet.
Expected answers stay outside native inference. No Qwen/foreground controller,
forced words/audio/wakeup, edited owner recording or original held-out hash change.

After initial reused10-case behavior review, adoption still requires this full
fresh20 inventory, valid independent input/speech clocks, all actual worker
receipts, ordinary8/8, private4/4, clarify/cancel/quiet/pause2/2 each, no false action
or stale value, and measured latency/backlog/playout. Invalid new source audio is
frozen and unscored, never rerolled or treated as a passing subset. A declaration
is not execution or qualification evidence. This preserves the distinction between
native engineering/data approval, reused development review and fresh acceptance.


## Full24 speech/control AND semantic pass; actual conversion failure reproduced

7544794 passes all24 and exact same-WAV/raw-ASR/words/times/receipts control,70 CPU
tests pass.7544795 obtains24/24 actual gpt-6-astra semantic passes, then conversion
rejects001 and004 because delegation precedes the complete WAV file end. It is
not a200ms emission-grid failure. Keep the rejected partial output unapproved.

001 actual delegation5.822835s, first clipend5.85s, last measured question word4.54s;
late waveform tail is quiet.004 actual delegation5.538613s, fileend6.9s, last
measured question word3.8s, but another acoustic burst occurs5.8–6.9s. Existing
GPT-Live user transcript identifies Laurel South by4.6s and contains no later user
word before10s. No assumption that the late burst is harmless is accepted without
an independent isolated-tail control. Pinned raw SHA and windowed CPU ASR keep
original audio, text, receipts and folds untouched. No expected words enter ASR.
Only after this read-only diagnosis choose an explicit measured-question clock
or repair a diagnosed raw TRAIN waveform in a new directory. No guard/threshold
weakening, changed original, passing subset or native optimizer submission.


## Complete-question word clock, retaining the entire original waveform

7545367 independent CPU full/prefix source004 ASR both locate the COMPLETE
Laurel South question through3.8s; actual GPT-Live user transcription identifies
South by4.6s, before delegation5.538613s. Isolated-window ASR can hallucinate:001
near-silent tail becomes Thank you. No isolated-tail output is promoted as a
label, and the original late004 acoustic burst is kept, not trimmed or attenuated.

The conversion checker used enclosing WAV clipend as question completion. New
native24-only policy checks every source question word against the independently
measured word sequence exactly, with valid intervals and complete project identity
for clarification. It gates actual delegation against the last measured question
word. Missing/reordered identity or an earlier delegation fails. Original file-end
metadata and actual delegation times/markers are untouched. Legacy old25 behavior
uses its original guard; no old baseline/corpus is rewritten.

Real24 replay now passes with maximum50ms grid error, still below the unchanged
200ms bar.28 focused tests pass. Full24 speech/control7544794 and actual Astra
semantic7544795 already pass; conversion-only wrapper will reverify every stored
input/response/hash and create a new version, without duplicate semantic requests,
new teacher sessions, source/heldout audio edits or optimizer updates. Rejected
partial22-case conversion is preserved. All native approval stages remain required.


## All native25 captures complete; propagated quality stamp policy

Remaining21 capture7544803 completes successfully; full25 CPU7544805 is running,
FP32math7544806 remains afterok.73 CPU tests7545523 pass. Conversion7545524 creates
all24 full measured-question timelines, then fail-closes at its final quality
stamp for022: generator still passes raw spelling WER0.095238 to stamp_batch,
instead of the explicit finite same-word canonical score0 already independently
audited on unchanged words/times/audio. Failed output remains immutable/unapproved.

New measured_speech_quality uses agreement_wer's verified declared policy, stores
raw WER and the exact finite mapping beside canonical WER, and is recomputed by
conversion validation. Legacy reports without the opt-in policy retain raw WER
behavior. Repeats, missing words, synonyms, wrong values and early timing remain
failures; the0.08 bar is unchanged.29 focused tests pass, including the actual
quality-row field consumed by stamp_batch. New versioned full24 conversion must
pass before encoding. Existing24 Astra responses are reused only through exact
hash/input verification; no new API requests or source recordings are required.


## Complete native input25 and conversion24; FP32 result retained

7544805 independently approves all25 actual native teacher inputs. FP327544806
executes every25 case and fails maximum0.002780318260192871 against unchanged
0.001; causal prefix error0.0 throughout. Baseline report SHA256
b890476073688ea418ec9fff834c2534a909fac647671444d6ca9282c6e373c6 retained and
backed with data. Existing higher-precision reference7545975 uses FP32 weights/
activations and FP64 linear arithmetic ONLY for numerical audit; same0.001 limit,
all25 required, no production/training alteration and no optimizer.

CPU7545751 passes74tests; full24 conversion7545753 now passes every quality stamp,
complete measured-question timing, exact speech/semantic evidence and all24 records.
Codec7545905 and scope7545907 pass. Complete24 capture7545908 (cap3), independent
input7545909 and FP32math7545910 form the exclusive new native approval chain from
source a32b165ade593e62f92c675872ba4b0c789450ce. Rejected partial/full conversions
remain immutable. No repeated teacher/semantic requests, source/heldout mutations,
optimizer submissions or behavioral/adoption claims. Both complete corpora need
independent native numerical-v2 approval before prepared eight-update pilot.


## Fresh20 input implementation; external generation approval pending

05ae360113829366768b7022684133b0618beac6 adds isolated user-only generation,
raw sample/receipt/schedule verification, same-audio counterfactual reuse, and
independent full20 CPU speech/time/peak audit at unchanged WER0.05/peak0.995.
CPU7546173 PASSES79tests including all five initial fresh-input mutation controls.
Further approval-reader checks recompute WER from hash-bound actual ASR, require
all20, and expose only waveform/routing plus worker-only world to inference;
expected answers, scripts and scoring references never enter the model payload.

Automatic approval review REJECTED the chained generation submission before any
command ran: the36 authored synthetic questions would be sent to OpenAI speech
API without explicit owner approval for that transfer. An asynchronous question
is pending. ZERO real generation requests or generation jobs were submitted;
CPU7546173 uses fake local audio only. Do not reroute around the rejection.
Native running audits and backup work continue independently. No fresh20 model
inference adapter/job or qualification result exists yet.


## Isolated fresh native inference path prepared, not executed

Full20 approval-reader CPU7546191 passes81tests. Separate
context_spanning_fresh_probe.py keeps existing TRAIN/development probe and native
backend files byte-identical. It reuses audited native Engine/model loading,
accepted checkpoint voice, persona and seed; no controller, response forcing,
wakeup, ret bias, timing/gain patch or TRAIN recorder. Fixed complete input
approval supplies only actual user waveform and routing; private worlds remain
worker-only and scoring references/scripts stay outside inference. Native RET
and arrived-audio backend behavior is reused unchanged. Observe-only records
unnecessary RET markers as evidence rather than executing unauthorized requests.

Prepared complete40 baseline7510331/candidate comparisons (20 identical user
stimuli each) require a committed all10-case native development review, fixed
independently restored eight-update candidate, all20 source quality, and at most
four total GPUs. No fresh inference or extra compute purchase submitted. Source
quality is not task qualification; every report remains quality_qualified=false
until full behavior/speech/causality assessment. External audio generation remains
paused pending the owner's API-transfer approval after automatic-review rejection.


## All25 numerical-v2 pass; complete24 input pass; gated eight-update pilot

7545975 COMPLETES14m09 and passes all25 at max0.000882565975189209 under0.001,
all causal-prefix errors0.0. Failed FP327544806 retained, no training precision
or numerical threshold change. Report SHA256
 ef01b6d875c2346f71ac03b7e6b38f1bd7dbec4e333c55dbb94dac6ed6678fd6.
All24 captures7545908 complete and full CPUinput7545909 PASS13s; FP32math7545910
running. Required full24 numerical-v2 reference7546608 queued afterany5910, runtime
requires full original baseline regardless verdict; incomplete result fail-closes.
Native pilot7546645 is queued afterok6608 from c93ac93, CPU7546529 PASS85tests.
ZERO optimizer updates at submission. It rechecks both complete corpus chains,
warms from native520 only, resets optimizer for8updates, retains all28 histories,
all25 old teacher words/audio/receipts and KL1/hiddenMSE10. Separate full state
restore is mandatory inside job. All10 unforced development cases must be reviewed
before extension/adoption. Full20 fresh qualification remains prepared/unexecuted;
real speech generation paused awaiting explicit transfer approval after auto-review.
Backups154files/290599979bytes independently size/SHA256 verified11:56:05UTC.


## Declared first8 exposure, not full-corpus learning

Deterministic sampling from the retained28-component order and approved24-teacher
scope reaches000/008/016/020/001/009/017/021 during the first8 updates. Quiet014
first appears at27 and pause015 at31; all24 new teachers are covered by31. Dated
NATIVE-V2-EXPOSURE JSON binds the actual parent/scope/sampling hashes and marks
this as nonexecution evidence. Eight updates therefore prove mechanics/regression
only. Do not infer quiet/pause learning or full24 exposure from input approval or
a falling loss. No longer training is authorized by this calculation: first require
independent full optimizer/RNG restore and review every10 native development case.


## Native24 original FP32 result and matched quiet/pause baseline

FP327545910 completes all24 then FAILS10m35: only017 and006 numerical comparisons
exceed0.001, maximum0.0014115571975708008; ALL causal-prefix errors0.0. No source
word/audio failure is implied by this numerical verdict. Original full24 report,
data, log and pilot/test receipts backed10files/153814bytes, all size/SHA256 match.
Numerical-v2 reference7546608 is running and8-update7546645 remains dependent.

A two-case fixed baseline quiet014/pause015 replay is prepared using spare
allocated GPUs, no API access. Native520 weights/voice/protocol/seed and identical
repaired source PCM match planned candidate development cases. Complete raw
recordings, unnecessary RET markers and independent CPU speech clocks are kept;
quiet/pause behavior is still unqualified. Every output is evaluation-only and
must never become TRAIN labels. No held-out mutation, gain change or extra compute
purchase. Four old ordinary baseline results already exist; do not duplicate them.


## Native520 actual quiet/pause baseline FAIL2/2, independent word clocks

7547007_0/1 COMPLETED2m21/2m40. Quiet014: Which project should I check for now
spoken8.77–10.37s, seven words inside unchanged0–12.41s quiet interval and before
cue start9.76s. Pause015: Sure I am quiet spoken20.93–21.73s within unchanged
17.54–22.10s mandatory pause; afterwards supplies unrelated close-door/turn-off-light
instructions rather than required close-the-cupboard continuation. Two unnecessary
native RET markers observed, none executed; external API calls0. Independent ASR
matches actual native text WER0.0212766/0.0, valid restored word bounds/physical
onsets. All real user samples processed. Full JSON binds report/audio/source/ASR
hashes, exact word intervals and p95 compute/backlog; candidate not yet evaluated.

Baseline persona's unconditional Ask which project if the user has not identified
it may contribute to irrelevant questions. This is an observed wording confound,
not controlled causal evidence or authorization to rewrite approved prefix/history.
Keep original persona/approved teacher histories for candidate comparison. All
baseline outputs are evaluation-only, never TRAIN labels.31files/94949512bytes,
complete raw audio/history/ASR/receipts/logs, independently hash-verified12:13:02UTC.


## Both numerical corpora pass; pilot and complete10 native review chain

7546608 COMPLETES10m07, all24 pass max0.00043702125549316406 under unchanged
0.001, causal0.0 throughout. Original failed full24 FP327545910 max0.001411557
retained inside reference report and separate backup. All25 reference7545975 also
passes max0.000882566. Production/training precision and threshold remain unchanged.
Pilot7546645 is running sourcec93ac93 with all old28/25/KL1/hidden10 unchanged;
independent fresh-process full optimizer/RNG restore is inside same job. Complete10
native evaluation7547174 afterok6645 and independent CPU ASR7547175 afterany7174
queued using exclusive receipt. No completed updates, native behavioral improvement,
extension or adoption claimed at12:21UTC. New reference5files/178679bytes verified;
source bundle756cdfb complete-history verified. Actual future results must update
this ledger, not inherit a passing claim from source/input/math/mechanics approval.


## Actual8 updates completed and portable full state preserved

7546645 update process PASS8/8 in15.961207563988864seconds; frozen base/speech
weights unchanged, peak17747770880bytes, zero past/audio loss targets. Adapter
SHA256 df113fc5bf67365062830737fb2b2ad5f6715afe555e10e185272e19987e9759;
full optimizer/Python/NumPy/Torch/CUDA state SHA256
47b52ef890ecf64fa7c4cab9d5c1cf52bc74227d2bf676a13d9414ee2da95b4a. Parent's
exact28-component data hash preserved, old25+new24 teacher policies and KL1/hidden10
confirmed from stored report. Actual first8 sampling matches declaration exactly;
quiet/pause teachers not exposed yet. Independent fresh-process restore still
pending; complete10 native evaluation and actual speech review remain mandatory.
Critical11files/31207417bytes hash-backed12:24:35UTC. No extension/adoption or
behavior qualification inferred. Same7510331/accepted-voice ordinary baselines
7530265–68 already exist and must not be duplicated. New baseline quiet/pause
FAIL2/2 stands independently; conditional persona contribution remains unproven.


## Full independent state restore passes; native review running

7546645 COMPLETED13m09. Fresh-process full adapter/optimizer/Python/NumPy/Torch/
CUDA restoration PASS with all28 valid logits exact, report SHA256
869dd68ffcd5adf9e207709b9cf186f6b5a6c6accaaef3206950480b972b6fbb. Full completed
16-file31744462-byte run independently backed, critical snapshot unchanged.
Development7547174_0/1 starts12:36:18, remaining8 pending, full10 CPUASR7547175
queued. No extension or adoption. Same-source, same-voice baseline4ordinary
7530265–68 already available, valid independent speech WER0.0, so not resubmitted.

Slurm initially could not place2GPU jobs and reported no start estimate. Six
observe-only0–5 need no ASR GPU; pending requests changed in place to1GPU/110G,
consistent with successful historical ordinary probes. Four real-delegation6–9
remain2 distinctGPUs. Throttle2/max4GPUs unchanged, original receipt preserved,
no new/cancelled/replacement job or model/input/check change. First CLI attempt
used sbatch-style memory suffix; Slurm rejected it without changes. Corrected
numeric112640MiB request succeeded and jobs started. Both receipts versioned and
backed. [Official scontrol reference](https://slurm.schedmd.com/scontrol.html)
documents individual array-task updates and memory values in MiB. This administrative
resource change is not a native behavior fix, threshold relaxation or qualification.


## All10 native outputs complete; fresh speech explicitly approved

12:51UTC:7547174_0–9 complete, independent full10 CPU speech7547175 running.
Preliminary raw text: ice regresses vs baseline; quiet/pause speech, invented private
numbers, cancellation failure. Final behavior review requires actual WAV, independent
ASR and worker clocks, not raw text. No extension/adoption. First6 actual outputs
78files229195823bytes verified12:48:14UTC; full10 backup in progress.
Owner explicit answer Approve synthetic speech generation resolves prior automatic
transfer rejection. Fresh generation7548247, independent all20 quality7548248;
exclusive receipt native-fresh-inputs-local-20261019.json fixes36 synthetic user
utterances,20caseplan and sourcec93ac93. No owner conversations/private world/scoring
answers exported. No fresh inference; invalid input freezes whole qualification,
no reroll or passing subset. LOCAL active, no cloud purchase/validation/submissions.


## Zero-update complete objective gradient diagnostic planned

Native8 candidate7546645 is rejected for extension/adoption on independently
confirmed ice regression and task/audio failures. New explicit gradient-audit
phase retains native520 parent, exact28 actual components,25 reapproved native
histories,24 new approved teachers, KL1/hidden10 and every objective/weight. It
computes separate gradients for ALL28+4reference+25old+24new=81 cases without an
optimizer step or checkpoint. Aggregates are weighted by the exact declared8/32
sampling schedule; negative cosine diagnoses first-order objective conflict at a
fixed parent, not Adam attribution or native behavior. Existing training/reload
path defaults unchanged. CPU arithmetic/schedule/guard tests required before the
zero-update GPU diagnostic. No repair, learning-rate change or extension chosen.


## Full native review rejects candidate; full fresh20 source freezes unscored

All10 actual native runs7547174 complete, full10 independent CPU ASR7547175
complete6m14. Full review NATIVE-V2-COMPLETE10-REVIEW-2026-10-19.json rejects
7546645 extension/adoption. Actual read/cancel receipts do execute; native voice
still wrong/invented facts, quiet/pause failures, no cancellation confirmation.
Six speech-quality passes under unchanged0.08/valid bounds; four failures retained.
Full124 native files263114323bytes plus independent speech2files170049bytes backed.
Zero-update complete81 gradient diagnostic7548377 source60a7afe queued afterCPU
7548310 PASS87tests. No further optimizer submitted, all28/25/KL1/hidden10 preserved.

Fresh generation7548247 COMPLETE2m14, full20 CPU quality7548248 FAILED5m58.
18/20cases pass,2fail:001colour/color rawWER0.05555556;002warm/worn WER0.08333333,
all physical waveform bars/bounds pass. Whole20 freezes UNSCORED; no inference,
passing subset, repeated speech generation or TRAIN use. All159 source/audio/
receipt/quality/log artifacts44064884bytes size/SHA256 verified12:58:45UTC.
Plan/scoring targets/audio/receipts unchanged. A separately versioned full20 local
large-v3 ASR control is prepared: no reference-text prompt, same0.05/0.995 bars,
raw WER retained, explicit color/colour spelling only; warm/worn NEVER equivalent.
This is DIAGNOSTIC ONLY, cannot satisfy existing input reader or authorize inference.
It may distinguish a recognizer error from source mispronunciation, without rerolling
or changing original failed verdict. Only turbo currently cached; official package's
pinned large-v3 weight would be downloaded by local cluster ASR, no speech API calls.

Timing correction: prior status headings12:51/13:00 were rounded forward by a few
minutes. Exact receipt times, Slurm clocks and backup-verification UTC timestamps
are authoritative. This note and every new receipt use actual UTC clocks.


## Proposed complete24 exposure in bounded8 updates, not executed

Existing candidate's first8 updates sample only8of24 teachers; quiet014 and pause015
first appear27/31 under the original schedule. Pure proposed coverage helper averages
4private/4simple or2clarify/2cancel teachers on each respective existing kind visit.
All24 are therefore covered exactly once in first8; old actual28/old25/reference
sampling and total new-objective kind weights remain unchanged. Two CPU tests verify
all24/quiet/pause across four8-update blocks and reject missing/misclassified teachers
or wrong retention clock. Actual parent manifests generate explicit declaration
NATIVE-V2-PROPOSED-COMPLETE24-EXPOSURE-2026-10-19.json; optimizer updates observed0,
training is NOT wired to helper or approved by this declaration. No learning rate
chosen while zero-update full81 diagnostic7548377 is pending. Distinguish coverage
hypothesis from training execution and useful native behavior.


## Read-only waveform and teacher-coverage observations

Raw native4/5 PCM rendered with ffmpeg showwavespic and original interval shading:
quiet burst inside0–12.41s and pause burst inside17.54–22.10s are visually present.
JSON review binds plot hashes, exact source duration, annotation intervals and raw
WAV SHA256; no audio/gain/clock edit and no claim of human listening. Independent
invalid ASR tails remain failed, not used as a passing subset.
Approved old25 consists19private/4simple/1clarify/1cancel. The simple teachers are
recovery008–011; ordinary science probes012–015 are absent. Native520's correct
science is thus retained generalization, not explicitly rehearsed in those25
teacher targets. This observation does not establish a regression cause or justify
new labels. Preserve all25/actual28/KL1/hidden10, keep all science checks unchanged.


## Opt-in complete24 pilot and exact-state continuation guard prepared, not launched

The declared coverage helper is now connected only behind --native-v2-full24-per8.
It averages exact half-groups(4private/simple,2clarify/cancel) per existing kind
visit, keeps new objective weight1 and old25/actual28/KL1/hidden10/old sampling
unchanged, records every teacher view/weight and fails if any24 are missing or
repeated in an8-update block. All old default paths remain opt-in separated.
Explicit learning-rate choices1e-5/2e-6 are recorded in full24 policy and exact
optimizer state metadata; rate NOT CHOSEN, no optimizer submitted. Full81 gradient
7548377 and same-audio full20 ASR7548438 still queued at preparation snapshot.

A future32-additional-update continuation is possible only for an independently
restored8-update FULL24 pilot, same immutable Git source/recipe/rate, exact full
optimizer/RNG state and a COMMITTED clean-checkout full10 native review binding
candidate checkpoint/state/reload, every raw report/WAV and complete independent
speech audit. All4 ordinary cores and their strict independent speech must pass;
every other native task, failed quality and actual receipt requires explicit
review/reason. This permits explicitly reviewed bounded research, never adoption
or passing-subset qualification. Rejected legacy7546645 cannot pass full24 guard.
No review exists authorizing such continuation. All25 approved histories/targets,
reference KL1/hidden10 and all28 actual components are preserved. No new labels.
Native development/fresh wrappers accept8 or properly resumed40 with complete
restore, but retain all10/40 cases and all behavioral/quality review prerequisites.

13 focused fixture tests pass locally: coverage, config/retention guards, complete
review binding, altered/missing evidence, ordinary regressions, actual failed-task
research review, committed Git document checks and existing trained-checkpoint
verification. These are fixture/source checks, not native numerical or behavior
acceptance. Full independent CPU suite must pass before a new optimizer.


## Diagnostics now execute; held replacement handshake prevents duplicates (2026-10-19 13:28:20 UTC)

Full sourcecf28726 independent CPU7548658 PASS101tests2.891s. Gradient diagnostic
workq7548377 and same-audio ASR GPU7548438 remained pending with no start estimate.
Official short-debug policy rechecked: one zero-update exploratory native gradient
audit is eligible, normal training/corpus jobs remain workq. Old gradient held;
held replacement7548671 verified effective interactive reservation/partition, old
cancelled0elapsed before replacement released. Exact payload60a7afe unchanged,
new wrappercf28726;1GPU/110G/30min, existing allocation1.5x credits, no purchase.
7548671 starts13:25:31UTC. All81 required/no optimizer/no new checkpoint.

ASR oldGPU7548438 held, CPU replacement7548672 held/verified hasNO GPU request,
old cancelled0elapsed, replacement released after independent source101tests.
Same large-v3 model/full20/all38 clips/no expected-text prompting; raw failed
quality/frozen unscored set unchanged. CPU downloading pinned official weights.
Always use squeue --all including hidden partition to audit all active jobs.
Transitions have separate immutable receipts; original submission records retained.
Both migration receipts + source/test evidence4files39794bytes independently
size/SHA256 verified13:27:04UTC. Sourcecf28726 complete-history bundle verified
SHA256fc3d3c2c5dd368877c3838b8c232be271867c0f0b1e7d4b260988c3f72cefe08.
[Official short-debug guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm-advanced/).
No learning rate selected/new optimizer or extension submitted; native/whole-system FAIL.
