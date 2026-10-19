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
