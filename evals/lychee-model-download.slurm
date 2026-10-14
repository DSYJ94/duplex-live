#!/bin/bash
#SBATCH --job-name=dl-lychee-assets
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dl-lychee-assets-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo" \
  --env HF_HOME="$DL/models/hf-cache" --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" python -u -m evals.lychee_model_download \
  --vendor "$DL/vendor/lychee-fd-7bb1068c59b578e1f8e65fb762cd26708c14cc6d" \
  --out "$DL/runs/lychee-assets-${SLURM_JOB_ID}"
