#!/bin/bash
#SBATCH --job-name=dl-moshi-data
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-moshi-data-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
git rev-parse HEAD
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH="$DL/pylibs-new:$DL/repo" --env HF_HOME="$DL/models/hf-cache" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$DL/pytorch-new.sif" python -u -m evals.moshi_prepare_data \
    --batch "$DL/runs/synth-scale1-q/audio" --out "$DL/runs/moshi-data-${SLURM_JOB_ID}"
