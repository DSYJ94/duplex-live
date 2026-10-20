#!/bin/bash
#SBATCH --job-name=dl-native-audio49-native10
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-audio49-native10-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}" "${NATIVE_AUDIO49_CHAIN_RECEIPT:?}" "${NATIVE_V2_CANDIDATE_JOB:?}"
cd "$DL_TASK_SOURCE"
test "$(git rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git status --porcelain)"
trap 'RC=$?; python3 -m evals.native_audio49_chain preserve --receipt "$NATIVE_AUDIO49_CHAIN_RECEIPT" --stage native10 --job "$SLURM_JOB_ID" --exit-code "$RC" || true' EXIT
export DL_SESSION_SOURCE="$DL_TASK_SOURCE" DL_SESSION_COMMIT="$DL_TASK_COMMIT"
RC=0
bash evals/native-audio49-session.sh || RC=$?
# Audit all original10 attempts even if a native case failed; never select a subset.
# The durable server watcher submits the complete speech audit after this job exits.
exit "$RC"
