#!/bin/bash
#SBATCH --job-name=dl-old25-timing-cache
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-old25-timing-cache-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
: "${OLD25_TIMING_REPAIR_JOB:?Complete25 timing approval required}"
[[ "$OLD25_TIMING_REPAIR_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
SOURCE="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
run() {
 apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$SOURCE/moshi:$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 --env HF_HOME="$DL/models/hf-cache" --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" -u "$@"
}
run -m unittest evals.harness.test_native_old25_timing_cache
run -m evals.native_old25_timing_cache --data "$DL/runs/native-old25-timing-data-$OLD25_TIMING_REPAIR_JOB" \
 --out "$DL/runs/native-old25-codes-$SLURM_JOB_ID"
