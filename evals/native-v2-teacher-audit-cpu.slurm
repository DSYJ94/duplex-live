#!/bin/bash
#SBATCH --job-name=dl-native-v2-teacher-audit-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-native-v2-teacher-audit-cpu-%j.out
set -euo pipefail
export NATIVE_ASR_DEVICE=cpu
# Same all24 gates and original clock; explicit CPU Whisper FP32 measurement.
: "${DL_TASK_SOURCE:?Immutable source required}"
source "$DL_TASK_SOURCE/evals/native-v2-teacher-audit.slurm"
