#!/bin/bash
#SBATCH --job-name=dl-native-v2-semantic
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-v2-semantic-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
: "${NATIVE_TEACHER_JOB:?Raw native teacher receipt required}"
: "${NATIVE_TEACHER_AUDIT_JOB:?Measured teacher audit required}"
[[ "$NATIVE_TEACHER_JOB" =~ ^[0-9]+$ && "$NATIVE_TEACHER_AUDIT_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
run() {
 apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" -u "$@"
}
CHECKS="$DL/runs/native-v2-teacher-audit-$NATIVE_TEACHER_AUDIT_JOB/teacher_checks.json"
SEMANTIC="$DL/runs/native-v2-semantic-${SLURM_JOB_ID}"
run -m evals.native_v2_semantic_review --batch "$DL/runs/native-v2-teacher-$NATIVE_TEACHER_JOB" \
 --checks "$CHECKS" --out "$SEMANTIC"
run -m evals.recovery_teacher_convert --checks "$CHECKS" --semantic-approval "$SEMANTIC/semantic-approval.json" \
 --out "$DL/runs/native-v2-teacher-data-${SLURM_JOB_ID}"
