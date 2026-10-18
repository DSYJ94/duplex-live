#!/bin/bash
#SBATCH --job-name=dl-fresh-hybrid-quality
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-fresh-hybrid-quality-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
: "${DUPLEX_FRESH_INPUTS_JOB:?Fresh generated inputs required}"
[[ "$DUPLEX_FRESH_INPUTS_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-annotation-7379026/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u -m evals.duplex_fresh_inputs judge \
 --data "$DL/runs/duplex-fresh-inputs-$DUPLEX_FRESH_INPUTS_JOB" \
 --out "$DL/runs/duplex-fresh-quality-${SLURM_JOB_ID}"
