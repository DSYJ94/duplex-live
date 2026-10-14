#!/bin/bash
#SBATCH --job-name=dl-recovery-repair
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-recovery-repair-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" -u \
  -m evals.recovery_repair --source "$DL/runs/recovery-curriculum-7406244" \
  --out "$DL/runs/recovery-repaired-${SLURM_JOB_ID}"
