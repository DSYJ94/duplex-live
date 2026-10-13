#!/bin/bash
#SBATCH --job-name=dl-moshi-models
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dl-moshi-models-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
git rev-parse HEAD
apptainer exec --bind /projects,/lus,/home \
    --env PYTHONPATH="$DL/pylibs-new:$DL/repo" --env HF_HOME="$DL/models/hf-cache" \
    "$DL/pytorch-new.sif" python -u -m evals.moshi_model_download \
    --out "$DL/runs/moshi-models-${SLURM_JOB_ID}"
