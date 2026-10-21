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
