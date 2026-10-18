#!/bin/bash
#SBATCH --job-name=dl-qwen35-kernels
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-qwen35-kernels-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
: "${QWEN35_RUNTIME_JOB:?Verified isolated runtime required}"
[[ "$QWEN35_RUNTIME_JOB" =~ ^[0-9]+$ ]]
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env HF_HUB_DISABLE_XET=1 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
 --env HF_HOME="$DL/models/hf-cache" \
 "$DL/pytorch-new.sif" python -u -m evals.qwen35_kernels \
 --runtime "$DL/runs/qwen35-runtime-$QWEN35_RUNTIME_JOB" --out "$DL/runs/qwen35-kernel-assets-${SLURM_JOB_ID}"
