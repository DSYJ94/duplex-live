#!/bin/bash
#SBATCH --job-name=dl-native-precision-no-tf32-watch
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-native-precision-no-tf32-watch-%j.out
set -euo pipefail
ulimit -c 0
: "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}" "${NATIVE_PRECISION_CHAIN_RECEIPT:?}"
cd "$DL_TASK_SOURCE"
test "$(git rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1 DL_CHAIN_WATCHER_JOB="$SLURM_JOB_ID"
python3 -u -m evals.native_precision_chain --receipt "$NATIVE_PRECISION_CHAIN_RECEIPT"
