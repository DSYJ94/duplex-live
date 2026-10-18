#!/bin/bash
#SBATCH --job-name=dl-qwen35-worker-contract
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:12:00
#SBATCH --output=logs/dl-qwen35-worker-contract-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
: "${QWEN35_ASSETS_JOB:?Pinned CPU asset proof required}"
[[ "$QWEN35_ASSETS_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
EXTRA=()
if [ "${QWEN_COMPACT_REPLIES:-0}" = 1 ]; then EXTRA+=(--compact-replies); fi
if [ "${QWEN_SPOKEN_OUTPUT_CONTRACT:-0}" = 1 ]; then EXTRA+=(--spoken-output-contract); fi

if [ -n "${QWEN35_KERNEL_ASSETS_JOB:-}" ]; then
 [[ "$QWEN35_KERNEL_ASSETS_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--kernel-assets "$DL/runs/qwen35-kernel-assets-$QWEN35_KERNEL_ASSETS_JOB")
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" python -u -m evals.qwen_response_executor \
 "${EXTRA[@]}" --controller-transport "${QWEN_CONTROLLER_TRANSPORT:-in-process}" --pending-speech-protocol "${QWEN_PENDING_SPEECH_PROTOCOL:-array}" --provider qwen35 --attention-backend "${QWEN_CONTROLLER_ATTENTION_BACKEND:-math}" \
 --assets "$DL/runs/qwen35-controller-assets-$QWEN35_ASSETS_JOB/assets.json" \
 --out "$DL/runs/qwen35-worker-contract-${SLURM_JOB_ID}"
