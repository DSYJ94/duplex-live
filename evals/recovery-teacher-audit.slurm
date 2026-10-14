#!/bin/bash
#SBATCH --job-name=dl-teacher-audit
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-teacher-audit-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
  -m evals.recovery_teacher_quality --batch "$DL/runs/recovery-teacher-7408766" \
  --out "$DL/runs/recovery-teacher-audit-${SLURM_JOB_ID}"
