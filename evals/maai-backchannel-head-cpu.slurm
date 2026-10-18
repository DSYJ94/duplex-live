#!/bin/bash
#SBATCH --job-name=dl-bcdet-head-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/dl-bcdet-head-cpu-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
: "${MAAI_ASSETS_JOB:?Pinned tensor inventory required}"
[[ "$MAAI_ASSETS_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
 "$DL/pytorch-new.sif" python -u -m evals.maai_backchannel_head \
 --inventory "$DL/runs/maai-backchannel-assets-$MAAI_ASSETS_JOB/report.json" \
 --out "$DL/runs/maai-backchannel-head-${SLURM_JOB_ID}"
