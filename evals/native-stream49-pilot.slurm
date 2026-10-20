#!/bin/bash
#SBATCH --job-name=dl-native-stream49-pilot
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/dl-native-stream49-pilot-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}" "${NATIVE_AUDIO49_PREFLIGHT_JOB:?}" "${NATIVE_AUDIO49_CHAIN_RECEIPT:?}"
cd "$DL_TASK_SOURCE"
test "$(git rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git status --porcelain)"
trap 'RC=$?; python3 -m evals.native_audio49_chain preserve --receipt "$NATIVE_AUDIO49_CHAIN_RECEIPT" --stage training --job "$SLURM_JOB_ID" --exit-code "$RC" || true' EXIT
export NATIVE_OLD25_CODES_JOB=7543730 NATIVE_OLD25_SCOPE_JOB=7543889 NATIVE_OLD25_APPROVAL_JOB=7544805 NATIVE_OLD25_MATH_JOB=7545975
export NATIVE_V2_CODES_JOB=7545905 NATIVE_V2_SCOPE_JOB=7545907 NATIVE_V2_APPROVAL_JOB=7545909 NATIVE_V2_MATH_JOB=7546608
export NATIVE_V2_FULL24_PER8=1 NATIVE_V2_LEARNING_RATE=2e-6 NATIVE_V2_WARM_ORDINARY_WEIGHT=64 NATIVE_V2_WARM_ORDINARY_APPROVAL_JOB=7554946 NATIVE_V2_WARM_ORDINARY_CLOCK=1
export NATIVE_AUDIO49_AUDIT_JOB=7572398 NATIVE_V2_TRAIN_PHASE=train
unset NATIVE_AUDIO49_PREFLIGHT_ONLY NATIVE_AUDIO49_STREAM_GRADIENT_READINESS
export NATIVE_AUDIO49_STREAM_AUXILIARY=1 NATIVE_AUDIO49_COMPLETE_BATCH=1 NATIVE_AUDIO49_PARENT_ADAM=1 NATIVE_AUDIO49_STEP_SCALE=0.1
bash evals/native-v2-train.slurm
# The durable server watcher verifies completion and fresh restore before native10.
