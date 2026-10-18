#!/bin/bash
#SBATCH --job-name=dl-hybrid-speech-audit
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-hybrid-speech-audit-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
: "${DUPLEX_LIVE_JOB:?Complete concurrent source required}"
[[ "$DUPLEX_LIVE_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
EXTRA=()
if [ "${DUPLEX_HYBRID_ALLOW_PARTIAL_DIAGNOSTIC:-0}" = 1 ]; then EXTRA+=(--include-completed-cases-from-failed-suite); fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-annotation-7379026/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u -m evals.duplex_hybrid_audit \
 "${EXTRA[@]}" --run "$DL/runs/duplex-live-transport-$DUPLEX_LIVE_JOB" \
 --out "$DL/runs/duplex-live-transport-audit-${SLURM_JOB_ID}"
