#!/bin/bash
#SBATCH --job-name=dl-state-context
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=logs/dl-state-context-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
PHASE=${CONTROL_CONTEXT_PHASE:-preflight}
[[ "$PHASE" = preflight || "$PHASE" = full ]]
EXTRA=(--recipe evals/fixtures/state-context-plan-20261018.json)
: "${STATE_TEACHER_JOB:?Complete verified matched-state corpus required}"
[[ "$STATE_TEACHER_JOB" =~ ^[0-9]+$ ]]
if [ "$PHASE" = full ]; then
 : "${CONTROL_CONTEXT_PREFLIGHT_JOB:?Actual declared-update proof required}"
 [[ "$CONTROL_CONTEXT_PREFLIGHT_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--preflight "$DL/runs/controller-context-${CONTROL_CONTEXT_PREFLIGHT_JOB}")
fi
cd "$DL_TASK_SOURCE"
unset OPENAI_API_KEY
ARGS=(--phase "$PHASE" --data "$DL/runs/controller-state-teacher-$STATE_TEACHER_JOB"
 --assets "$DL/runs/qwen35-controller-assets-7512612/assets.json"
 --kernel-assets "$DL/runs/qwen35-kernel-assets-7513645"
 --out "$DL/runs/controller-context-${SLURM_JOB_ID}" "${EXTRA[@]}")
for MODE in train restore; do
 RESTORE=()
 if [ "$MODE" = restore ]; then RESTORE+=(--restore-only); fi
 apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 \
  --env OPENAI_API_KEY= \
  "$DL/pytorch-new.sif" python -u -m training.control.train_state_context "${ARGS[@]}" "${RESTORE[@]}"
done
