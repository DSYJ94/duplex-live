#!/bin/bash
#SBATCH --job-name=dl-qwen35-speech-control
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-qwen35-speech-control-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
EXTRA=()
if [ "${QWEN_PRIOR_ASSISTANT_HISTORY:-0}" = 1 ]; then EXTRA+=(--prior-assistant-history); fi
if [ -n "${DUPLEX_CONTROL_CAPTURED_JOB:-}" ]; then
 [[ "$DUPLEX_CONTROL_CAPTURED_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--captured-run "$DL/runs/duplex-hybrid-probe-$DUPLEX_CONTROL_CAPTURED_JOB")
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" python -u -m evals.qwen35_speech_control_probe "${EXTRA[@]}" \
 --assets "$DL/runs/qwen35-controller-assets-7512612/assets.json" \
 --kernel-assets "$DL/runs/qwen35-kernel-assets-7513645" \
 --out "$DL/runs/qwen35-speech-control-${SLURM_JOB_ID}"
