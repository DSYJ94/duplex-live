#!/bin/bash
#SBATCH --job-name=dl-live-access
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/dl-live-access-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
# Load the existing project credential inside the trusted cluster shell.
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" \
  -u -m evals.gpt_live_access_probe --out "$DL/runs/gpt-live-access-${SLURM_JOB_ID}/report.json"
