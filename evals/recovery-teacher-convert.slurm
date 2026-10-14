#!/bin/bash
#SBATCH --job-name=dl-teacher-convert
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-teacher-convert-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
: "${TEACHER_CHECKS:?Independent teacher check file required}"
export DL TEACHER_CHECKS
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
PY="$DL/envs/moshi-annotation-7379026/bin/python"
"$PY" -u -m evals.recovery_teacher_convert --checks "$TEACHER_CHECKS" \
  --out "$DL/runs/recovery-teacher-data-${SLURM_JOB_ID}"
"$PY" -u -m evals.recovery_compose_data --base "$DL/runs/recovery-data-7407554" \
  --teacher "$DL/runs/recovery-teacher-data-${SLURM_JOB_ID}" \
  --out "$DL/runs/recovery-mixed-data-${SLURM_JOB_ID}"
CONTAINER
