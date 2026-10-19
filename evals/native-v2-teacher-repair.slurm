#!/bin/bash
#SBATCH --job-name=dl-native-v2-teacher-repair
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-native-v2-teacher-repair-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
PARENT=7543257
AUDIT=7543297
EXTRA=()
if [ "${NATIVE_TEACHER_CLARITY_RETRY:-0}" = 1 ]; then
 PARENT=7543779
 AUDIT=7543781
 EXTRA+=(--clarity-retry)
fi
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" -u \
 -m evals.native_v2_teacher_repair --parent "$DL/runs/native-v2-teacher-$PARENT" \
 --checks "$DL/runs/native-v2-teacher-audit-$AUDIT/teacher_checks.json" \
 --out "$DL/runs/native-v2-teacher-$SLURM_JOB_ID" "${EXTRA[@]}"
