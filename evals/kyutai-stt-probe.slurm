#!/bin/bash
#SBATCH --job-name=dl-streaming-asr-probe
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:08:00
#SBATCH --output=logs/dl-streaming-asr-probe-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${KYUTAI_STT_ASSETS_JOB:?Pinned streaming ASR assets required}"
: "${KYUTAI_TTS_MIMI_LOAD_JOB:?Independent exact shared-codec load proof required}"
[[ "$KYUTAI_STT_ASSETS_JOB" =~ ^[0-9]+$ && "$KYUTAI_TTS_MIMI_LOAD_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" python -u -m evals.kyutai_stt_probe \
 --assets "$DL/runs/kyutai-stt-assets-$KYUTAI_STT_ASSETS_JOB/assets.json" \
 --mimi-load-approval "$DL/runs/kyutai-tts-mimi-load-$KYUTAI_TTS_MIMI_LOAD_JOB/report.json" \
 --data "$DL/runs/recovery-data-7407554" \
 --scenario recovery-train-008 --scenario recovery-train-019 --scenario recovery-train-020 \
 --out "$DL/runs/kyutai-stt-probe-${SLURM_JOB_ID}"
