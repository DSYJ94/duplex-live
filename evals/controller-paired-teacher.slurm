#!/bin/bash
#SBATCH --job-name=dl-controller-paired-teacher
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-controller-paired-teacher-%j.out
set -euo pipefail
export DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
EXTRA=()
if [ "${CONTROL_TEACHER_PREVIEW:-0}" = 1 ]; then EXTRA+=(--preview); fi
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
 "$DL/pytorch-new.sif" python -u -m training.control.paired_data "${EXTRA[@]}" \
 --out "$DL/runs/controller-paired-teacher-${SLURM_JOB_ID}"
