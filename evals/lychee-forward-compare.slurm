#!/bin/bash
#SBATCH --job-name=dl-lychee-forward
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-lychee-forward-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${LYCHEE_NATIVE_JOB:?Actual native forward capture required}"
[[ "$LYCHEE_NATIVE_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.lychee_forward_compare \
 --assets "$DL/runs/lychee-assets-7414609/manifest.json" \
 --native-run "$DL/runs/lychee-vllm-probe-$LYCHEE_NATIVE_JOB" \
 --out "$DL/runs/lychee-forward-compare-${SLURM_JOB_ID}"
