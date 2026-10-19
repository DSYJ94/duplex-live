#!/bin/bash
#SBATCH --job-name=dl-native-v2-approved-convert
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-native-v2-approved-convert-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
: "${NATIVE_TEACHER_AUDIT_JOB:?Complete24 independent speech approval required}"
: "${NATIVE_SEMANTIC_JOB:?Complete24 actual semantic approval required}"
[[ "$NATIVE_TEACHER_AUDIT_JOB" =~ ^[0-9]+$ && "$NATIVE_SEMANTIC_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" -u \
 -m evals.recovery_teacher_convert \
 --checks "$DL/runs/native-v2-teacher-audit-$NATIVE_TEACHER_AUDIT_JOB/teacher_checks.json" \
 --semantic-approval "$DL/runs/native-v2-semantic-$NATIVE_SEMANTIC_JOB/semantic-approval.json" \
 --out "$DL/runs/native-v2-teacher-data-$SLURM_JOB_ID"
# Reuses actual24/24 semantic responses only after re-verifying every unchanged
# input and response hash. No API requests, rerendering, waveform edits or training.
