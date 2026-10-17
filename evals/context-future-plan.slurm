#!/bin/bash
#SBATCH --job-name=dl-context-future
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=115000M
#SBATCH --time=00:05:00
#SBATCH --output=logs/dl-context-future-%A_%a.out
set -euo pipefail
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${DL_ELIGIBILITY:?CPU eligibility report required}"
: "${DL_ELIGIBILITY_SHA256:?Reviewed eligibility hash required}"
: "${DL_CAPTURE_MAP:?Source capture map required}"
: "${SLURM_ARRAY_TASK_ID:?Array task required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
MODE=()
if [ "${DL_NATIVE_RESPONSE_BOUNDARY:-0}" = 1 ]; then
 test "${DL_LATE_PRIVATE:-0}" = 0
 MODE+=(--ordinary-response-boundary)
fi
if [ "${DL_LATE_PRIVATE:-0}" = 1 ]; then MODE+=(--late-diagnostic); fi
FIELDS=$(python3 -m evals.context_future_select --eligibility "$DL_ELIGIBILITY" \
 --sha256 "$DL_ELIGIBILITY_SHA256" --map "$DL_CAPTURE_MAP" --index "$SLURM_ARRAY_TASK_ID" "${MODE[@]}")
mapfile -t SELECTED <<< "$FIELDS"
test "${#SELECTED[@]}" = 3
export CONTEXT_ACTUAL_REPLAY_JOBS="${SELECTED[0]}" CONTEXT_ACTUAL_SOURCE_COMMIT="${SELECTED[1]}"
export CONTEXT_ACTUAL_TEACHER_ID="${SELECTED[2]}"
export CONTEXT_ACTUAL_EARLY_WINDOW=1 CONTEXT_ACTUAL_FUTURE_COMPONENT=1
export CONTEXT_ACTUAL_NATIVE_RESPONSE_BOUNDARY="${DL_NATIVE_RESPONSE_BOUNDARY:-0}"
if [ "${DL_LATE_PRIVATE:-0}" = 1 ]; then export CONTEXT_ACTUAL_EARLY_WINDOW=0; fi
export CONTEXT_ACTUAL_ENGINE_REPLAY=1 CONTEXT_ACTUAL_RECORDED_VOICE=1
export CONTEXT_REQUEST_POLICY=native-prelead-epad
export CONTEXT_ACTUAL_NATIVE_AUDIO_FUTURE="${DL_NATIVE_AUDIO:-0}"
if [ "${DL_LATE_PRIVATE:-0}" = 1 ] && [ "$CONTEXT_ACTUAL_NATIVE_AUDIO_FUTURE" != 0 ]; then
 echo 'Late correction supports independently audited teacher acoustic conditioning only' >&2
 exit 1
fi
echo "Future task $SLURM_ARRAY_TASK_ID: history $CONTEXT_ACTUAL_REPLAY_JOBS / teacher $CONTEXT_ACTUAL_TEACHER_ID; native audio $CONTEXT_ACTUAL_NATIVE_AUDIO_FUTURE"
bash "$DL_TASK_SOURCE/evals/context-training-proof.slurm"
