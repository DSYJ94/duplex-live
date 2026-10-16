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
 if [ "${CONTEXT_ACTUAL_EARLY_WINDOW:-0}" = 1 ]; then ACTUAL+=(--actual-early-window); fi
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
 for J in $CONTEXT_ACTUAL_HEAD_JOBS; do
  [[ "$J" =~ ^[0-9]+$ ]]
  HEAD+=(--actual-head-data "$DL/runs/context-training-proof-$J")
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
