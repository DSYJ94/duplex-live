#!/bin/bash
#SBATCH --job-name=dl-function-name-grammar
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-function-name-grammar-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
unset OPENAI_API_KEY
EXTRA=()
if [ -n "${FUNCTION_GRAMMAR_CANDIDATE_JOB:-}" ]; then
 [[ "$FUNCTION_GRAMMAR_CANDIDATE_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--candidate "$DL/runs/controller-context-$FUNCTION_GRAMMAR_CANDIDATE_JOB")
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" python -u -m evals.function_name_probe "${EXTRA[@]}" \
 --assets "$DL/runs/qwen35-controller-assets-7512612/assets.json" \
 --kernel-assets "$DL/runs/qwen35-kernel-assets-7513645" \
 --runs "$DL/runs" \
 --out "$DL/runs/function-name-probe-${SLURM_JOB_ID}"
