#!/bin/bash
#SBATCH --job-name=dl-recovery-data
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-recovery-data-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
export RECOVERY_CONVERSION_OUT="${RECOVERY_CONVERSION_OUT:-$DL/runs/recovery-data-${SLURM_JOB_ID}}"
export RECOVERY_BATCH="${RECOVERY_BATCH:-$DL/runs/recovery-curriculum-7406244}"
git rev-parse HEAD
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  --env HF_HOME="$DL/models/hf-cache" --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
  -m evals.recovery_prepare_data \
  --batch "$RECOVERY_BATCH" \
  --out "$RECOVERY_CONVERSION_OUT"
