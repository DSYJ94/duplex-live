#!/bin/bash
#SBATCH --job-name=dl-lychee-semantic-proof
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-lychee-semantic-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${LYCHEE_TRAINING_INPUTS_JOB:?Verified native target assembly required}"
[[ "$LYCHEE_TRAINING_INPUTS_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
STATES=()
for J in 7452624 7452625 7452626 7452627 7452628 7452629 7452630 7452631; do
 STATES+=("$DL/runs/lychee-vllm-probe-$J")
done
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.lychee_semantic_train \
 --assets "$DL/runs/lychee-assets-7414609/manifest.json" \
 --data "$DL/runs/lychee-training-inputs-$LYCHEE_TRAINING_INPUTS_JOB" \
 --source-data "$DL/runs/recovery-data-7407554" --history-roots "${STATES[@]}" \
 --codec-data "$DL/runs/lychee-codec-data-7452500" --codec-audit "$DL/runs/lychee-codec-audit-7452502" \
 --steps 8 --out "$DL/runs/lychee-semantic-proof-${SLURM_JOB_ID}"
