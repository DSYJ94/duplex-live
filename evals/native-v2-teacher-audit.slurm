#!/bin/bash
#SBATCH --job-name=dl-native-v2-teacher-audit
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-v2-teacher-audit-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
: "${NATIVE_TEACHER_JOB:?Raw native teacher receipt required}"
[[ "$NATIVE_TEACHER_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 --env HF_HOME="$DL/models/hf-cache" --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
 -m evals.recovery_teacher_quality --batch "$DL/runs/native-v2-teacher-$NATIVE_TEACHER_JOB" \
 --out "$DL/runs/native-v2-teacher-audit-${SLURM_JOB_ID}"
# A completed diagnostic report is not a passing full-corpus approval.
python3 -c 'import json,sys; r=json.load(open(sys.argv[1]))["results"]; assert len(r)==24 and len({x["scenario_id"] for x in r})==24 and all(x["speech_and_fact_checks_pass"] for x in r), "All24 TRAIN teacher recordings must pass"' \
 "$DL/runs/native-v2-teacher-audit-${SLURM_JOB_ID}/teacher_checks.json"
