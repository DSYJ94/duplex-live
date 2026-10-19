#!/bin/bash
#SBATCH --job-name=dl-old25-native-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-old25-native-cpu-%j.out
set -euo pipefail
export NATIVE_OLD25_DEVICE=cpu
: "${DL_TASK_SOURCE:?Immutable source required}"
source "$DL_TASK_SOURCE/evals/native-old25-native-approval.slurm"
