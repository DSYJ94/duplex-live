#!/bin/bash
#SBATCH --job-name=dl-teacher-encode
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-teacher-encode-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
: "${RECOVERY_MIXED_DATA:?Approved composed conversion required}"
export DL RECOVERY_MIXED_DATA
# Encode only new teacher audio; do not recompute the frozen approved base.
bash evals/recovery-encode.slurm
apptainer exec --bind /projects,/lus,/home --env PYTHONPATH="$DL/repo" \
  --env PYTHONDONTWRITEBYTECODE=1 "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" -u \
  -m evals.recovery_compose_cache --base "$DL/runs/recovery-codes-7408223" \
  --teacher "$DL/runs/recovery-codes-${SLURM_JOB_ID}" --data "$RECOVERY_MIXED_DATA" \
  --out "$DL/runs/recovery-mixed-codes-${SLURM_JOB_ID}"
