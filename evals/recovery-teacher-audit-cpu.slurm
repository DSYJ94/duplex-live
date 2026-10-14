#!/bin/bash
#SBATCH --job-name=dl-teacher-audit-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-teacher-audit-cpu-%j.out
# Explicit CPU execution of the same independent judge; no silent fallback.
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
  -m evals.recovery_teacher_quality --device cpu --batch "$DL/runs/recovery-teacher-7408766" \
  --out "$DL/runs/recovery-teacher-audit-${SLURM_JOB_ID}"
