#!/bin/bash
#SBATCH --job-name=dl-tts-gpu-probe
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-tts-gpu-probe-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${KYUTAI_TTS_ASSETS_JOB:?Complete verified pinned assets required}"
[[ "$KYUTAI_TTS_ASSETS_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" python -u -m evals.kyutai_tts_probe \
 --assets "$DL/runs/kyutai-tts-assets-$KYUTAI_TTS_ASSETS_JOB/assets.json" \
 --out "$DL/runs/kyutai-tts-probe-${SLURM_JOB_ID}"
