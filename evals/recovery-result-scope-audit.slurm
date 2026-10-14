#!/bin/bash
#SBATCH --job-name=dl-result-scope
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-result-scope-%j.out
# CPU-only structural audit; no optimizer, no sealed test labels.
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" -u \
  -m evals.recovery_result_scope --cache "$DL/runs/recovery-mixed-codes-7411513" \
  --manifest "$DL/runs/moshi-models-7378305/model-manifest.json" \
  --out "$DL/runs/recovery-result-scope-${SLURM_JOB_ID}"
