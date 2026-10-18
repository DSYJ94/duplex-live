#!/bin/bash
#SBATCH --job-name=dl-context-generalization
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-context-generalization-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
EXTRA=()
if [ -n "${CONTEXT_EXPECTED_UPDATES:-}" ]; then
 [[ "$CONTEXT_EXPECTED_UPDATES" =~ ^[0-9]+$ ]]
 EXTRA+=(--expected-updates "$CONTEXT_EXPECTED_UPDATES")
fi
if [ -n "${CONTEXT_CANDIDATE_JOB:-}" ]; then
 [[ "$CONTEXT_CANDIDATE_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--candidate "$DL/runs/controller-context-$CONTEXT_CANDIDATE_JOB")
fi
MODULE=evals.context_generalization
OUT_PREFIX=context-generalization
if [ "${CONTEXT_FRESH_CONTROL:-0}" = 1 ]; then
 MODULE=evals.context_fresh_control
 OUT_PREFIX=context-fresh-control
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" python -u -m "$MODULE" "${EXTRA[@]}" \
 --assets "$DL/runs/qwen35-controller-assets-7512612/assets.json" \
 --kernel-assets "$DL/runs/qwen35-kernel-assets-7513645" \
 --out "$DL/runs/${OUT_PREFIX}-${SLURM_JOB_ID}"
