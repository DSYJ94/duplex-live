# Supervised execution update

Source/pipeline main9f8d722 is committed and pushed. The sole supervisor is running
in `runs/native-control3-approval-20261021-v1`, following existing generation v2;
it cannot launch a duplicate generation job or any GPU/optimizer work. It requires
ALL168 source clips and full independent verification before actual40 teachers,
complete teacher quality checks and independent semantic review. A zero audit CLI
exit alone cannot allow progress: full verifier stages are mandatory. Every failed
corpus remains preserved, with no partial approval. Local credentials stay on Mac.

The independent semantic module `evals/native_control_semantic.py` revalidates the
whole measured teacher corpus before sending only authored synthetic TRAIN content
to the existing semantic-review model. Full raw responses and exact input digests
are bound to all40 decisions. It cannot grant conversion/numerical or training
approval. Actual semantic requests remain pending; 25 local fixture tests passed.

First80 source clips have a separate fully read-back-hashed S3 backup; that snapshot
is partial, never a word approval. The native archive remains independently verified.
Source CPU ASR records four threads/float32 and per-clip progress. See HANDOFF.md and
machine execution record for exact live processes, paths, snapshots and next work.

# Execution update: local credential supplied

The owner configured the ignored0600 credential on21October. Exact speech model
access returned HTTP200. Source collector de11d6a is running the complete168-clip
plan in `runs/native-control3-source-20261021-v2`, using the isolated local ASR
Python environment. V1 stopped before any API call because its environment lacked
requests; it remains preserved. All earlier blank-credential statements below are
historical. Source/teacher/semantic/numerical approvals remain pending.

The added `evals/native_control_teacher_quality.py` binds ALL40 original teacher
streams, the complete source word audit, label-free pinned Whisper responses and
measured phase/quiet checks. It cannot grant semantic, conversion, numerical or
training approval. It remains untested on actual new teacher artifacts until those
exist. No recipe, model, original data or heldout changes are made by preparation.

# Native control repair preparation

The latest candidate7590052 remains rejected for adoption. No optimizer update,
teacher collection, speech API request or GPU purchase was made in this preparation.
LOCAL is the sole executor. All task compute remains off.

## Measured source findings

The read-only auditor verified151 distinct recipe-bound files for all49 teachers:
both cache manifests and ingestion receipts, all49 code tensors, conversation WAVs
and sidecars. Every hash matched the saved7590052 recipe. It does not decode the
cached tensor masks or reapprove them. See
[coverage evidence](NATIVE-CONTROL-COVERAGE-2026-10-21.json).

Only two teachers have explicit quiet-control metadata: TRAIN014/alloy and
TRAIN015/echo. The other47 lack that metadata; this does not prove zero implicit
control learning. In the approved teacher waveforms, assistant channel0 contains
exactly zero PCM samples and no aligned words in the complete measured quiet
windows:5.70–12.41s and17.14–22.10s. The corresponding native windows fully
contain83 and61 frames. These source intervals do not require audio repair.

The existing source audit establishes that the objective includes silence/control
targets and that each native auxiliary teacher appeared once in the eight-update
stage. Missing silence loss is not the diagnosis. Source integrity and coverage
also do not prove student behavior: all nine new control diagnostics failed.

Working hypotheses, to test independently rather than assume:

1. Narrow wording/cue/voice/state coverage: hold voice, cue, checkpoint and seed
   fixed while varying wording, then vary cue and voice separately. Compare the
   original approved input with the failed diagnostics, without calling either fresh.
2. Insufficient exposure or regression in the new stage: compare baseline520 and
   candidate7590052 on the same complete diagnostic and native10 inputs. Only an
   independently approved bounded pilot can test additional exposure causally.
3. Temporal credit assignment: recorded-call detachment preserves forward cache
   values but prevents full temporal backpropagation. A separately reviewed
   zero-update gradient comparison is required before changing this policy.

No cause is established by the source audit. No loss, precision, sampling or
native clock change is adopted.

## New versioned source collection

[The source plan](../evals/fixtures/native-control-repair-v3-source-20261021.json)
contains40 TRAIN proposals,168 authored utterances and7,953 characters:

- Eight no-release cases with10/20/30 seconds of terminal silence.
- Eight later-cue cases followed by a useful reasoning answer.
- Eight cases with distractors and near-miss words before the valid cue.
- Eight pause/resume cases that retain the unfinished second step.
- Eight cue-replacement cases where the old cue must no longer release speech.

Eight cue words, four instruction forms and the existing TRAIN voices alloy/echo
are varied. Existing held-out voices/folds, all original source audio and teacher
histories remain unchanged. No owner recording, transcript or cue value is used.
Exact user-turn comparison found no casefolded matches against the original native2
source or either full-task plan. This is not proof of semantic holdout independence.
None of the new cases is advertised as a fresh evaluation holdout.

No-release scenes include an ordinary spoken exchange before the quiet request.
That preserves spoken and clock target groups across each scene, which the existing
native-stream objective requires. There is no invented terminal assistant utterance
and no relaxation of that objective's guard.

The new raw-source assembler supports these terminal quiet intervals. It requires
immutable clips and receipts binding exact words, voice, model and WAV hash. It
rejects empty/nonfinite/clipped audio at the unchanged peak0.995 ceiling, performs
no gain change or resampling, and refuses output-directory reuse. Missing any scene
prevents the complete raw-source manifest. It writes no approved training index.

Six tests passed, including all40 actual fixture assemblies, terminal silent PCM,
unmodified source hashes, no partial-batch approval, and rejection of acknowledgments,
empty/clipped/mismatched audio. These are source-assembly tests using artificial
test signals; they provide no speech quality or model capability evidence.

Reproduce the local checks:

```sh
/Users/daryl/.venvs/duplex-live/bin/python -m unittest evals.harness.test_native_control_repair -v
/Users/daryl/.venvs/duplex-live/bin/python -m evals.native_control_coverage \
  --recipe docs/NATIVE-7590052-RECIPE-2026-10-21.json \
  --archive runs/cloud-export-20261016/tree \
  --out runs/local-migration-20261021/NEW-control-coverage.json
```

Use a new output filename. Source assembly, after real clips exist, is:

```sh
python -m data.synthetic.native_control_render \
  --plan evals/fixtures/native-control-repair-v3-source-20261021.json \
  --clips PATH_TO_NEW_RECEIPTED_CLIPS --out PATH_TO_NEW_RAW_SOURCE
```

## Execution boundary and next action

The local API credential is still blank; its ignored file exists with0600 mode.
Do not copy the key to Runpod, a source bundle, a receipt or chat. Actual source
generation and teacher/worker execution have not been submitted. Historical
native-v2 teacher, quality, semantic and conversion code is explicitly scoped to24
scenes and cannot silently approve this40-scene collection. New bindings and
independent whole-collection reviews are still required.

Fresh capacity reads list no A100/H100 in the saved volume's EUR-IS-1 data center.
RTX PRO6000 Blackwell Server Edition96GB is listed LOW at$2.49/hour there, but this
different architecture has not been validated for this workload. No Pod was created.
The count1 read does not prove count2 capacity. The existing fullnative10 harness
still requires two distinct actual GPUs and the real external worker. Do not fake
device IDs, drop cases or substitute canned worker answers. Original compute cap
is$25; reported four-Pod total is$19.5219472303288 and may settle. The conservative
full-wallclock envelope is about$20.09. Storage remains separate at$7/month.

Next work: configure the local credential, complete the new teacher/approval
bindings, then generate and independently qualify ALL40 source scenes and actual
teacher histories. Complete baseline/candidate native10+independent speech and
ordinary4/4 before choosing a pilot parent. Only after all approvals may a new
bounded repair stage be defined and run. Keep all28 actual histories,49 approved
teachers, KL1/hidden10, original154 objectives and held-out hashes. Full40 sealed
paired outputs remain required before adoption. No unchanged extension of the
failed candidate and no frontend keyword mute is authorized as a substitute.

## Follow-on quality implementation

The raw-source format is now `dl-native-control3-raw-source-v2`. It retains all
original utterance WAVs and provider receipts byte for byte inside the new source
directory. No external Mac clip path is needed to revalidate it. The read-only
validator reconstructs every source stem and quiet window from those clips; even
a rewritten manifest hash cannot authorize PCM absent from the original clips.
Unexpected receipt fields are rejected before copying, so headers/credentials
cannot be silently included in this format. No real raw source existed when this
new format was implemented; original generated/model audio remains untouched.

`evals/native_control_source_quality.py` implements a label-free independent
Whisper-turbo word audit and verifier for ALL40 scenes and168 clips. The fixed
weight hash is aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a.
WER remains at most0.05 and peak at most0.995. Exact authored source word positions
must also match, so a changed cue/premise cannot pass merely because a long clip
has low aggregate WER. Raw recognizer outputs and waveform/receipt hashes are bound
to the audit. No reference text is prompted into ASR, audio is not edited, and the
report grants no student training approval. This CLI has not run on real new audio.

After real clips have been generated and assembled, use a new audit directory:

```sh
python -m evals.native_control_source_quality --raw NEW_RAW_SOURCE \
  --weights PINNED_LOCAL_TURBO_PT --out NEW_SOURCE_AUDIT --device cpu
python -m evals.native_control_source_quality --raw NEW_RAW_SOURCE \
  --verify NEW_SOURCE_AUDIT/source-word-audit.json
```

`evals/native_control_teacher_checks.py` implements the next teacher phase-check
component. It requires measured source-word completion, complete actual playback
and observation evidence, raw teacher PCM, independent teacher ASR and a clean
session close. A short output WAV alone cannot prove an unreleased quiet period.
It rejects acknowledgments and non-silent PCM during quiet (including sound missed
by ASR), near-miss/old-cue releases, early answers, repeated completed steps, absent
resume, unnecessary delegation and invalid word times. It is a component, not a
whole-collection teacher approval or a recording client. Full artifact/ASR bindings,
actual capture, independent semantic review and native conversion remain unfinished.

Fourteen fixture tests now pass, including complete40 assembly/reconstruction and
complete40 artificial teacher phase checks. These use artificial signals and word
records, never actual model outputs or approved labels. The actual source-word ASR
and teacher capture remain UNRUN. No new API/GPU/training work was submitted.

The existing GPT-Live model and WebSocket start/audio/close protocol were checked
against the official [model page](https://developers.openai.com/api/docs/models/gpt-live-1)
and [WebSocket guide](https://developers.openai.com/api/docs/guides/voice-websockets).
This verifies documentation compatibility, not account access or a successful call.
The local API credential remains blank at the last check. See the machine-readable
[quality preparation record](NATIVE-CONTROL3-QUALITY-PREPARATION-2026-10-21.json).

## Actual recorder and raw transport bindings implemented

`evals/native_control_teacher.py` now provides the new40-scene raw teacher recorder.
It verifies the complete source-word audit before opening a connection and permits
two concurrent GPT-Live sessions. A local OS file lease rejects overlapping
collectors; an old lock filename alone does not prevent a new process. Output
directories are immutable, no failed recording is retried, and an incomplete
transport leaves the whole collection unapproved. Source generations and source
ASR must finish first; the recorder has not made a real API call.

The teacher protocol is `native-control3-teacher-v1`, with the authored TRAIN-only
ordinary references and generic quiet/pause/cue-change rules. Cue values and
conversation state arrive through user audio. These instructions construct labels;
they do not change the native foreground runtime or establish unaided reasoning.
The collector runs no worker for these control-only cases. Unexpected delegations
remain in the evidence and must fail teacher quality.

Every input packet retains its sample interval, hash and actual send time. Every
returned PCM byte is flushed to a raw spool before its event is logged. Transcripts,
closed-session usage and incomplete transport evidence are retained. The derived
WAV preserves unbuffered packet availability; it is reconstructed from raw packets
by `evals/native_control_teacher_artifacts.py`. Editing a WAV and its manifest hash
cannot conceal a changed/dropped packet. The validator covers ALL40 records, their
source/model/instruction bindings, original user PCM plus15 silent seconds, complete
actual input/close evidence, transcripts and exact output playout. Passing this
validator is transport integrity only, never a teacher-label or model approval.

Teacher phase checks now use the later of independently measured user-word
completion and actual availability of the packet covering those words. Late or
missing input packets cannot borrow the planned question clock. These checks
retain strict quiet and complete-answer requirements; there is no response grace,
output mute or wording substitution.

Once the local credential, complete source generation and source-word audit exist:

```sh
python -m evals.native_control_teacher --raw NEW_RAW_SOURCE \
  --source-audit NEW_SOURCE_AUDIT/source-word-audit.json --out NEW_RAW_TEACHERS \
  --env-file runs/local-migration-20261021/.private/openai-backend.env
```

The credential is read locally from the ignored0600file; it is never copied into
artifacts or to Runpod. The actual CLI was checked with the blank local file and
stopped before source processing, output-directory creation or any connection.
Twenty-one artificial-fixture tests pass. Their simulated sockets/PCM are not
API access, authentic speech, labels or native qualification. See the
[teacher preparation record](NATIVE-CONTROL3-TEACHER-PREPARATION-2026-10-21.json).

Current empirical blocker: the local credential remains blank after three
consecutive preparation stages. Further source/teacher experiments need it.
Independent real teacher ASR, whole-collection semantic review and native conversion
still need completion and testing on actual artifacts. All numerical, retained
baseline/candidate complete10 and sealed40 gates remain required. No training,
source/data repair, adoption, new GPU or additional spending was performed here.
