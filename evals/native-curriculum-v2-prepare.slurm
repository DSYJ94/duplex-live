#!/bin/bash
#SBATCH --job-name=dl-native-source-v2-audit
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-native-source-v2-audit-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable committed source required}"
: "${DL_TASK_COMMIT:?Immutable source identity required}"
: "${NATIVE_SOURCE_JOB:?Source generation receipt required}"
[[ "$NATIVE_SOURCE_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
SOURCE="${NATIVE_SOURCE_DIR:-$DL/runs/native-curriculum-v2-$NATIVE_SOURCE_JOB}"
cmp "$SOURCE/specifications.json" "$DL_TASK_SOURCE/evals/fixtures/native-curriculum-v2-20261018.json"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HOME="$DL/models/hf-cache" --env OMP_NUM_THREADS=8 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
 -m evals.recovery_prepare_data --native-v2-train-scope --batch "$SOURCE" \
 --out "$DL/runs/native-curriculum-v2-data-${SLURM_JOB_ID}"
