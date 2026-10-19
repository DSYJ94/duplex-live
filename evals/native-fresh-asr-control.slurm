#!/bin/bash
#SBATCH --job-name=dl-native-fresh-asr-control
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-native-fresh-asr-control-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source identity required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-annotation-7379026/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
 -m evals.native_fresh_asr_control --data "$DL/runs/native-fresh-inputs-7548247" \
 --prior "$DL/runs/native-fresh-judge-7548248/quality.json" \
 --out "$DL/runs/native-fresh-asr-control-$SLURM_JOB_ID"
# Fixed stronger local ASR; no reference prompts, new speech calls or input approval.
