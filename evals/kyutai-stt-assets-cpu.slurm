#!/bin/bash
#SBATCH --job-name=dl-stt-assets-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-stt-assets-cpu-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${KYUTAI_TTS_COMPAT_JOB:?Successful isolated package import probe required}"
[[ "$KYUTAI_TTS_COMPAT_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=0 --env HF_HUB_DISABLE_XET=1 \
 "$DL/pytorch-new.sif" python -u -m evals.kyutai_stt_assets \
 --compat "$DL/runs/kyutai-tts-cpu-compat-$KYUTAI_TTS_COMPAT_JOB" \
 --out "$DL/runs/kyutai-stt-assets-${SLURM_JOB_ID}"
