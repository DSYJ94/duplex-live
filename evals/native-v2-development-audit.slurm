#!/bin/bash
#SBATCH --job-name=dl-native-v2-development-asr
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-native-v2-development-asr-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable committed source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
: "${NATIVE_V2_DEVELOPMENT_ARRAY_JOB:?Complete ten-case native development array required}"
[[ "$NATIVE_V2_DEVELOPMENT_ARRAY_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
RUNS=()
for I in $(seq 0 9); do RUNS+=("$DL/runs/native-v2-development-${NATIVE_V2_DEVELOPMENT_ARRAY_JOB}_$I"); done
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 --env HF_HOME="$DL/models/hf-cache" --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
 -m evals.moshi_speech_audit --runs "${RUNS[@]}" --device cpu \
 --trim-leading-silence --trim-trailing-silence --keep-silent-failures \
 --out "$DL/runs/native-v2-development-asr-$SLURM_JOB_ID"
# Retain silent/bad clips as failures. Every raw clip, word clock, task/receipt and
# packet/backlog metric still needs independent review; this is no qualification stamp.
