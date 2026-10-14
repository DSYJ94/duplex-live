#!/bin/bash
#SBATCH --job-name=dl-live-replay
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/dl-live-replay-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" \
  -u -m evals.gpt_live_replay --batch "$DL/runs/synth-scale1-q/audio" \
  --out "$DL/runs/gpt-live-replay-${SLURM_JOB_ID}/t1"
