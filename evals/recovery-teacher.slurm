#!/bin/bash
#SBATCH --job-name=dl-recovery-teacher
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-recovery-teacher-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" -u \
  -m evals.recovery_teacher --data "$DL/runs/recovery-data-7407554" \
  --out "$DL/runs/recovery-teacher-${SLURM_JOB_ID}"
