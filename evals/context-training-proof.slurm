#!/bin/bash
#SBATCH --job-name=dl-context-train-proof
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --output=logs/dl-context-train-proof-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
ARGS=(--assets "$DL/runs/context-spanning-assets-7454205/manifest.json"
 --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
 --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav"
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json"
 --cache "$DL/runs/recovery-mixed-codes-7411513"
 --cache "$DL/runs/recovery-codes-7434926" --cache "$DL/runs/recovery-codes-7435538"
 --out "$DL/runs/context-training-proof-${SLURM_JOB_ID}")
if [ "${CONTEXT_TEXT_PADDING:-legacy-epad0}" = context-pad3 ]; then ARGS+=(--text-padding context-pad3); fi
if [ "${CONTEXT_REQUEST_POLICY:-source-dispatch}" != source-dispatch ]; then
 ARGS+=(--request-policy "$CONTEXT_REQUEST_POLICY")
fi
if [ "${CONTEXT_INTEGRATED_HISTORY:-0}" = 1 ]; then ARGS+=(--history-audit); fi
if [ "${CONTEXT_HISTORY_LOGITS:-0}" = 1 ]; then ARGS+=(--history-logits); fi
if [ "${CONTEXT_PILOT:-0}" = 1 ]; then
 ARGS+=(--steps 128 --pilot-preflight "$DL/runs/context-training-proof-${CONTEXT_PREFLIGHT_JOB:-7455157}"
  --layout-audit "$DL/runs/context-training-proof-${CONTEXT_LAYOUT_AUDIT_JOB:-7455458}/history-audit.json")
fi
run_container() {
 apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" python -u "$@"
}
if [ "${CONTEXT_CODEC_AUDIT:-0}" = 1 ]; then
 run_container -m evals.context_codec_audit \
  --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
  --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
  --cache "$DL/runs/recovery-mixed-codes-7411513" --out "$DL/runs/context-codec-audit-${SLURM_JOB_ID}"
 exit 0
fi
if [ -n "${CONTEXT_ACTUAL_REPLAY_JOBS:-}" ]; then
 : "${CONTEXT_ACTUAL_SOURCE_COMMIT:?Actual collector commit required}"
 ACTUAL=()
 if [ "${CONTEXT_ACTUAL_TRAINED_COLLECTOR:-0}" = 1 ]; then ACTUAL+=(--actual-trained-collector); fi
 if [ "${CONTEXT_ACTUAL_CANCEL_ACK:-0}" = 1 ]; then ACTUAL+=(--actual-cancel-ack); fi
 if [ "${CONTEXT_ACTUAL_NATIVE_AUDIO_FUTURE:-0}" = 1 ]; then ACTUAL+=(--actual-native-audio-future); fi
 if [ -n "${CONTEXT_ACTUAL_TEACHER_ID:-}" ]; then ACTUAL+=(--actual-teacher-id "$CONTEXT_ACTUAL_TEACHER_ID"); fi
 if [ "${CONTEXT_ACTUAL_EARLY_WINDOW:-0}" = 1 ]; then ACTUAL+=(--actual-early-window); fi
 if [ "${CONTEXT_ACTUAL_NATIVE_RESPONSE_BOUNDARY:-0}" = 1 ]; then ACTUAL+=(--actual-native-response-boundary); fi
 if [ "${CONTEXT_ACTUAL_FUTURE_COMPONENT:-0}" = 1 ]; then ACTUAL+=(--actual-future-component); fi
 if [ "${CONTEXT_ACTUAL_ENGINE_REPLAY:-0}" = 1 ]; then ACTUAL+=(--actual-engine-replay); fi
 if [ "${CONTEXT_ACTUAL_RECORDED_VOICE:-0}" = 1 ]; then ACTUAL+=(--actual-recorded-voice); fi
 if [ "${CONTEXT_ACTUAL_NATIVE_PRIMING:-0}" = 1 ]; then ACTUAL+=(--actual-native-priming); fi
 if [ "${CONTEXT_ACTUAL_COLD_GRAPHS:-0}" = 1 ]; then ACTUAL+=(--actual-cold-graphs); fi
 for J in $CONTEXT_ACTUAL_REPLAY_JOBS; do
  [[ "$J" =~ ^[0-9]+$ ]]
  ACTUAL+=(--actual-replay "$DL/runs/context-spanning-probe-$J")
 done
 run_container -m evals.context_training_proof "${ARGS[@]}" "${ACTUAL[@]}" \
  --actual-source-data "$DL/runs/recovery-data-7407554" \
  --actual-source-commit "$CONTEXT_ACTUAL_SOURCE_COMMIT"
 exit 0
fi
if [ -n "${CONTEXT_ACTUAL_HEAD_JOBS:-}" ]; then
 HEAD=()
 if [ "${CONTEXT_MIXED_TRAIN_DATA:-0}" = 1 ] && [ "${CONTEXT_EARLY_TRAIN_DATA:-0}" = 1 ]; then
  echo 'Mixed and early-only component modes are mutually exclusive' >&2
  exit 1
 fi
 for J in $CONTEXT_ACTUAL_HEAD_JOBS; do
  [[ "$J" =~ ^[0-9]+$ ]]
  if [ "${CONTEXT_MIXED_TRAIN_DATA:-0}" = 1 ]; then
   HEAD+=(--actual-mixed-data "$DL/runs/context-training-proof-$J")
  elif [ "${CONTEXT_EARLY_TRAIN_DATA:-0}" = 1 ]; then
   HEAD+=(--actual-early-data "$DL/runs/context-training-proof-$J")
  else
   HEAD+=(--actual-head-data "$DL/runs/context-training-proof-$J")
  fi
 done
 HEAD+=(--actual-source-data "$DL/runs/recovery-data-7407554")
 if [ "${CONTEXT_ACTUAL_TEMPORAL_AUDIT:-0}" = 1 ]; then
  run_container -m evals.context_training_proof "${ARGS[@]}" "${HEAD[@]}" --actual-temporal-audit
  exit 0
 fi
 if [ "${CONTEXT_ACTUAL_TEMPORAL_FIT:-0}" = 1 ]; then
  : "${CONTEXT_ACTUAL_TEMPORAL_PREFLIGHT_JOB:?Passed matching temporal audit required}"
  [[ "$CONTEXT_ACTUAL_TEMPORAL_PREFLIGHT_JOB" =~ ^[0-9]+$ ]]
  HEAD+=(--actual-temporal-fit --actual-temporal-preflight "$DL/runs/context-training-proof-$CONTEXT_ACTUAL_TEMPORAL_PREFLIGHT_JOB")
  run_container -m unittest evals.harness.test_context_training_state
  if [ -n "${CONTEXT_RESUME_TRAINING_JOB:-}" ]; then
   [[ "$CONTEXT_RESUME_TRAINING_JOB" =~ ^[0-9]+$ ]]
   HEAD+=(--resume-training "$DL/runs/context-training-proof-$CONTEXT_RESUME_TRAINING_JOB/training-state.pt")
  fi
  if [ -n "${CONTEXT_WARM_START_JOB:-}" ]; then
   [[ "$CONTEXT_WARM_START_JOB" =~ ^[0-9]+$ ]]
   HEAD+=(--warm-start-proof "$DL/runs/context-training-proof-$CONTEXT_WARM_START_JOB")
  fi
  HEAD+=(--actual-fit-updates "${CONTEXT_ACTUAL_FIT_UPDATES:-8}")
  HEAD+=(--actual-sampling "${CONTEXT_ACTUAL_SAMPLING:-component-order}")
  HEAD+=(--actual-retention-sampling "${CONTEXT_ACTUAL_RETENTION_SAMPLING:-legacy-modulo}")
  HEAD+=(--actual-response-first-word-weight "${CONTEXT_ACTUAL_RESPONSE_FIRST_WORD_WEIGHT:-1}")
 fi
 run_container -m evals.context_training_proof "${ARGS[@]}" "${HEAD[@]}"
 run_container -m evals.context_training_proof "${ARGS[@]}" "${HEAD[@]}" --reload
 exit 0
fi
run_container -m unittest evals.harness.test_context_training_loss
if [ "${CONTEXT_TEACHER_AUDIT:-0}" = 1 ]; then
 AUDITS=()
 for J in ${CONTEXT_TEACHER_AUDIT_JOBS:-7455157 7455712}; do
  [[ "$J" =~ ^[0-9]+$ ]]
  AUDITS+=(--teacher-audit "$DL/runs/context-training-proof-$J")
 done
 run_container -m evals.context_training_proof "${ARGS[@]}" "${AUDITS[@]}"
 exit 0
fi
if [ "${CONTEXT_HISTORY_AUDIT:-0}" = 1 ]; then
 run_container -m evals.context_training_proof "${ARGS[@]}" --history-audit-only
 exit 0
fi
run_container -m evals.context_training_proof "${ARGS[@]}"
run_container -m evals.context_training_proof "${ARGS[@]}" --reload
