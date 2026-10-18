#!/bin/bash
#SBATCH --job-name=dl-live-owner-trial
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:30:00
#SBATCH --output=logs/dl-live-owner-trial-%j.out
set -euo pipefail
export DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
export DL_LIVE_RUN="$DL/runs/duplex-live-owner-trial-$SLURM_JOB_ID"
EXTRA=()
if [ "${DUPLEX_CONFIRMED_PAUSE:-0}" = 1 ]; then EXTRA+=(--lexical-confirmed-pause); fi
if [ "${QWEN_COMPACT_REPLIES:-0}" = 1 ]; then EXTRA+=(--compact-replies); fi
if [ "${DUPLEX_WARM_LONG_PREFILL:-0}" = 1 ]; then EXTRA+=(--warm-long-prefill); fi
if [ "${DUPLEX_CLIENT_PAUSE:-0}" = 1 ]; then EXTRA+=(--include-speech-context --client-pause-protocol); fi
if [ "${DUPLEX_SEMANTIC_PAUSE:-0}" = 1 ]; then EXTRA+=(--semantic-pause); fi
if [ "${DUPLEX_BOUND_RESUME:-0}" = 1 ]; then EXTRA+=(--bound-resume-target); fi
if [ "${DUPLEX_CANCEL_RESUME:-0}" = 1 ]; then EXTRA+=(--allow-cancel-resume); fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" python -u -m serving.hybrid.server "${EXTRA[@]}" \
 --stt-assets "$DL/runs/kyutai-stt-assets-7510852/assets.json" \
 --tts-assets "$DL/runs/kyutai-tts-assets-7510636/assets.json" \
 --mimi-load-approval "$DL/runs/kyutai-tts-mimi-load-7510722/report.json" \
 --controller-assets "$DL/runs/qwen35-controller-assets-7512612/assets.json" \
 --kernel-assets "$DL/runs/qwen35-kernel-assets-7513645" \
 --out "$DL_LIVE_RUN" --host 0.0.0.0 --port 8788 --max-sessions 10 --idle-seconds 14400
