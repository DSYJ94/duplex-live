#!/bin/bash
#SBATCH --job-name=dl-lychee-preflight
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-lychee-preflight-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$DL/repo" \
  --env HF_HUB_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" python -u -m evals.lychee_checkpoint_preflight
