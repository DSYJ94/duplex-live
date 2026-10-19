#!/bin/bash
#SBATCH --job-name=dl-old25-timing-repair
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=logs/dl-old25-timing-repair-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
run() {
 apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  --env HF_HOME="$DL/models/hf-cache" --env OPENAI_API_KEY= \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u "$@"
}
OUT="$DL/runs/native-old25-timing-repair-$SLURM_JOB_ID"
run -m unittest evals.harness.test_native_old25_timing_repair
run -m evals.native_old25_timing_repair --out "$OUT"
run -m evals.native_old25_timing_repair --convert-approval "$OUT/report.json" \
 --out "$DL/runs/native-old25-timing-data-$SLURM_JOB_ID"
# This full 25-history conversion is not cache/input/numerical or optimizer approval.
