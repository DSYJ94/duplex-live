#!/bin/bash
#SBATCH --job-name=dl-controller-head-train
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-controller-head-train-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
: "${CONTROL_TEACHER_DATA_JOB:?Complete full teacher corpus required}"
: "${CONTROL_BASELINE_JOB:?Complete actual frozen baseline required}"
[[ "$CONTROL_TEACHER_DATA_JOB" =~ ^[0-9]+$ ]]
[[ "$CONTROL_BASELINE_JOB" =~ ^[0-9]+$ ]]
cd "$DL_TASK_SOURCE"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" python -u -m training.control.train_head \
 --assets "$DL/runs/qwen35-controller-assets-7512612/assets.json" \
 --kernel-assets "$DL/runs/qwen35-kernel-assets-7513645" \
 --data "$DL/runs/controller-text-teacher-$CONTROL_TEACHER_DATA_JOB" \
 --baseline "$DL/runs/controller-text-baseline-$CONTROL_BASELINE_JOB" \
 --out "$DL/runs/controller-head-train-${SLURM_JOB_ID}"
