#!/bin/bash
#SBATCH --job-name=dl-context-assets
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dl-context-assets-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env PYTHONDONTWRITEBYTECODE=1 \
 "$DL/pytorch-new.sif" python -u -m evals.context_spanning_download \
 --vendor "$DL/vendor/context-spanning-79df302fa2820cae10c61bb2b7f302c52edb6ed2" \
 --out "$DL/runs/context-spanning-assets-${SLURM_JOB_ID}"
