#!/bin/bash
#SBATCH --job-name=dl-future-speech-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-future-speech-cpu-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${CONTEXT_FUTURE_JOBS:?Completed native-audio futures required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
FUTURES=()
for J in $CONTEXT_FUTURE_JOBS; do
 [[ "$J" =~ ^[0-9]+$ ]]
 FUTURES+=("$DL/runs/context-training-proof-$J")
done
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=16 \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u -m evals.moshi_speech_audit \
 --future-runs "${FUTURES[@]}" --device cpu --trim-trailing-silence --keep-silent-failures \
 --out "$DL/runs/context-future-speech-audit-${SLURM_JOB_ID}"
