#!/bin/bash
#SBATCH --job-name=dl-lychee-concurrent-audit
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-lychee-concurrent-audit-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${LYCHEE_CONCURRENT_JOB:?Actual concurrent capture required}"
[[ "$LYCHEE_CONCURRENT_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
OUT="$DL/runs/lychee-concurrent-audit-${SLURM_JOB_ID}"
mkdir "$OUT"
export DL_AUDIT_OUT="$OUT" DL_AUDIT_CAPTURE="$DL/runs/lychee-concurrent-$LYCHEE_CONCURRENT_JOB"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" /bin/bash -c '
 set -euo pipefail
 PY=/projects/u6si/duplex-live/envs/moshi-annotation-7379026/bin/python
 "$PY" -u -m evals.lychee_utterance_export --capture "$DL_AUDIT_CAPTURE" --out "$DL_AUDIT_OUT/utterances"
 mapfile -t CLIPS < "$DL_AUDIT_OUT/utterances/audit-runs.txt"
 "$PY" -u -m evals.moshi_speech_audit --runs "${CLIPS[@]}" --out "$DL_AUDIT_OUT/asr" --trim-trailing-silence --keep-silent-failures
 '
