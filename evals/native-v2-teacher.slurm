#!/bin/bash
#SBATCH --job-name=dl-native-v2-teacher
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-v2-teacher-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
: "${NATIVE_SOURCE_AUDIT_JOB:?Measured source approval receipt required}"
[[ "$NATIVE_SOURCE_AUDIT_JOB" =~ ^[0-9]+$ ]]
PROTOCOL="${NATIVE_TEACHER_PROTOCOL:-native-dialogue-v2}"
[[ "$PROTOCOL" = native-dialogue-v2 || "$PROTOCOL" = native-dialogue-v3 ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" -u \
 -m evals.recovery_teacher --data "$DL/runs/native-curriculum-v2-data-$NATIVE_SOURCE_AUDIT_JOB" \
 --out "$DL/runs/native-v2-teacher-${SLURM_JOB_ID}" --scope native-v2 --protocol "$PROTOCOL"
