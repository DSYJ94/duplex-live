#!/bin/bash
#SBATCH --job-name=dl-live-fresh
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-live-fresh-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" \
  -u -m evals.gpt_live_replay --batch "$DL/runs/recovery-repaired-7407535" \
  --conversion "$DL/runs/recovery-data-7407554" --scenario recovery-validation-000 \
  --out "$DL/runs/gpt-live-fresh-${SLURM_JOB_ID}/t1"
