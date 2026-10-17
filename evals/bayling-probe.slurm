#!/bin/bash
#SBATCH --job-name=dl-bayling-probe
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=115000M
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-bayling-probe-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${BAYLING_ASSETS_JOB:?Verified assets required}"
: "${BAYLING_RUNTIME_JOB:?Passing CPU runtime required}"
: "${BAYLING_SCENARIO:?Approved validation scene required}"
[[ "$BAYLING_ASSETS_JOB" =~ ^[0-9]+$ && "$BAYLING_RUNTIME_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/bayling-runtime-$BAYLING_RUNTIME_JOB:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.bayling_probe \
 --assets "$DL/runs/bayling-assets-$BAYLING_ASSETS_JOB/manifest.json" \
 --runtime "$DL/runs/bayling-runtime-$BAYLING_RUNTIME_JOB/runtime.json" \
 --data "$DL/runs/recovery-data-7407554" --scenario "$BAYLING_SCENARIO" \
 --out "$DL/runs/bayling-probe-${SLURM_JOB_ID}"
