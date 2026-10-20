#!/bin/bash
#SBATCH --job-name=dl-native-audio49-speech10
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-audio49-speech10-%j.out
set -euo pipefail
ulimit -c 0
: "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}" "${NATIVE_AUDIO49_CHAIN_RECEIPT:?}" "${NATIVE_V2_DEVELOPMENT_ARRAY_JOB:?}"
cd "$DL_TASK_SOURCE"
test "$(git rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git status --porcelain)"
trap 'RC=$?; python3 -m evals.native_audio49_chain preserve --receipt "$NATIVE_AUDIO49_CHAIN_RECEIPT" --stage speech10 --job "$SLURM_JOB_ID" --exit-code "$RC" || true' EXIT
bash evals/native-v2-development-audit.slurm
# No autonomous extension, adoption or fresh20 run is authorized by speech quality.
