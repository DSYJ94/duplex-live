#!/bin/bash
#SBATCH --job-name=dl-context-capture
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-context-capture-%A_%a.out
set -euo pipefail
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${DL_CAPTURE_PLAN:?Approved-teacher capture plan required}"
: "${DL_CAPTURE_PLAN_SHA256:?Reviewed plan hash required}"
: "${SLURM_ARRAY_TASK_ID:?Array task required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
FIELDS=$(python3 -m evals.context_capture_select --plan "$DL_CAPTURE_PLAN" \
 --sha256 "$DL_CAPTURE_PLAN_SHA256" --index "$SLURM_ARRAY_TASK_ID")
mapfile -t SELECTED <<< "$FIELDS"
test "${#SELECTED[@]}" = 3
export CONTEXT_SCENARIO="${SELECTED[0]}" CONTEXT_EXPECTED_KIND="${SELECTED[1]}"
export CONTEXT_PRIVATE_VALUE="${SELECTED[2]}" CONTEXT_ASSETS_JOB=7454205
export CONTEXT_VOICE=accepted-synthetic CONTEXT_MATH_ATTENTION=1
export CONTEXT_BACKEND_MODE=native-asr CONTEXT_TRAIN_HISTORY=1
# Match the native protocol, zero wrappers and exact voice codes used by replay
# and the actual-history trainer. A released-protocol capture is not interchangeable.
export CONTEXT_ADAPTER_JOB=7463825 CONTEXT_ADAPTER_ZERO=1 CONTEXT_ADAPTER_DISABLED=0
export CONTEXT_LAYOUT_AUDIT_JOB=7463825
echo "Capture task $SLURM_ARRAY_TASK_ID: $CONTEXT_SCENARIO / $CONTEXT_EXPECTED_KIND; plan $DL_CAPTURE_PLAN_SHA256"
bash "$DL_TASK_SOURCE/evals/context-spanning-probe.slurm"
