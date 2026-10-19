#!/bin/bash
#SBATCH --job-name=dl-native-fresh2-asr-control-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-native-fresh2-asr-control-cpu-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source identity required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
: "${NATIVE_FRESH_INPUTS_JOB:?Complete sealed V2 inputs required}"
: "${NATIVE_FRESH_QUALITY_JOB:?Complete failed V2 source quality required}"
[[ "$NATIVE_FRESH_INPUTS_JOB" =~ ^[0-9]+$ && "$NATIVE_FRESH_QUALITY_JOB" =~ ^[0-9]+$ ]]
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-annotation-7379026/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
 -m evals.native_fresh_asr_control_v2 --device cpu --data "$DL/runs/native-fresh2-inputs-$NATIVE_FRESH_INPUTS_JOB" \
 --prior "$DL/runs/native-fresh2-judge-$NATIVE_FRESH_QUALITY_JOB/quality.json" \
 --out "$DL/runs/native-fresh2-asr-control-$SLURM_JOB_ID"
# Fixed second local recognizer; no reference prompts, new speech calls or input approval.
