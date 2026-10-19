#!/bin/bash
#SBATCH --job-name=dl-native-fresh-asr-control-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-native-fresh-asr-control-cpu-%j.out
set -euo pipefail
: "${DL_TASK_SOURCE:?Immutable source required}"
export NATIVE_FRESH_ASR_DEVICE=cpu
# Bash ignores the shared payload's SBATCH GPU directive; this allocation is CPU only.
bash "$DL_TASK_SOURCE/evals/native-fresh-asr-control.slurm"
