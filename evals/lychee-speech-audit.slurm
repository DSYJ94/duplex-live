#!/bin/bash
#SBATCH --job-name=dl-lychee-speech-audit
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-lychee-speech-audit-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
TASK_SOURCE="${DL_TASK_SOURCE:-$DL/repo}"
if [ "$TASK_SOURCE" != "$DL/repo" ]; then
 : "${DL_TASK_COMMIT:?Immutable source commit required}"
 test "$(git -C "$TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
 test -z "$(git -C "$TASK_SOURCE" status --porcelain)"
fi
cd "$TASK_SOURCE"
: "${LYCHEE_AUDIT_RUNS:?Native audio artifact directory names required}"
RUNS=()
for NAME in $LYCHEE_AUDIT_RUNS; do
 [[ "$NAME" =~ ^lychee-[a-z0-9-]+$ ]]
 RUNS+=("$DL/runs/$NAME")
done
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u -m evals.moshi_speech_audit \
 --runs "${RUNS[@]}" --trim-trailing-silence --keep-silent-failures \
 --out "$DL/runs/lychee-speech-audit-${SLURM_JOB_ID}"
