#!/bin/bash
#SBATCH --job-name=dl-tts-mimi-load-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/dl-tts-mimi-load-cpu-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${KYUTAI_TTS_ASSETS_JOB:?Pinned assets required}"
[[ "$KYUTAI_TTS_ASSETS_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" python -u -m evals.kyutai_tts_mimi_load \
 --assets "$DL/runs/kyutai-tts-assets-$KYUTAI_TTS_ASSETS_JOB/assets.json" \
 --out "$DL/runs/kyutai-tts-mimi-load-${SLURM_JOB_ID}"
