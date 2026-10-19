#!/bin/bash
#SBATCH --job-name=dl-native-v2-asr-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-v2-asr-cpu-%j.out
set -euo pipefail
export NATIVE_ASR_DEVICE=cpu
# Explicit CPU Whisper measurement, recorded in report.json; no GPU is requested.
: "${DL_TASK_SOURCE:?Immutable source required}"
source "$DL_TASK_SOURCE/evals/native-v2-asr-alignment-control.slurm"
