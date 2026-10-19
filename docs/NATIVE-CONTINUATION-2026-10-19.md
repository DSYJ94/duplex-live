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
