#!/bin/bash
#SBATCH --job-name=dl-lychee-runtime-deps
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-lychee-runtime-deps-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
TASK_SOURCE="${DL_TASK_SOURCE:-$DL/repo}"
if [ "$TASK_SOURCE" != "$DL/repo" ]; then
 : "${DL_TASK_COMMIT:?Immutable worktree commit required}"
 test "$(git -C "$TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
 test -z "$(git -C "$TASK_SOURCE" status --porcelain)"
fi
cd "$TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 "$DL/pytorch-new.sif" \
 python -u -m evals.lychee_vllm_dependencies \
 --source "$DL/vendor/lychee-fd-7bb1068c59b578e1f8e65fb762cd26708c14cc6d" \
 --target "$DL/envs/lychee-vllm-deps-${SLURM_JOB_ID}" \
 --out "$DL/runs/lychee-vllm-deps-${SLURM_JOB_ID}"
