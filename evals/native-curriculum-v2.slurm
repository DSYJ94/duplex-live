#!/bin/bash
#SBATCH --job-name=dl-native-source-v2
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=logs/dl-native-source-v2-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable committed source required}"
: "${DL_TASK_COMMIT:?Immutable source identity required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" \
 -u -m data.synthetic.recovery_curriculum \
 --specifications "$DL_TASK_SOURCE/evals/fixtures/native-curriculum-v2-20261018.json" \
 --out "$DL/runs/native-curriculum-v2-${SLURM_JOB_ID}"
