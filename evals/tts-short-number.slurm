#!/bin/bash
#SBATCH --job-name=dl-short-number-speech
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-short-number-speech-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
unset OPENAI_API_KEY
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env HF_HUB_OFFLINE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.tts_short_number_probe \
 --assets "$DL/runs/kyutai-tts-assets-7510636/assets.json" \
 --stt-assets "$DL/runs/kyutai-stt-assets-7510852/assets.json" \
 --mimi-load-approval "$DL/runs/kyutai-tts-mimi-load-7510722/report.json" \
 --out "$DL/runs/tts-short-number-${SLURM_JOB_ID}"
