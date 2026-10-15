#!/bin/bash
#SBATCH --job-name=dl-lychee-runtime-deps
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-lychee-runtime-deps-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 "$DL/pytorch-new.sif" \
 python -u -m evals.lychee_vllm_dependencies \
 --source "$DL/vendor/lychee-fd-7bb1068c59b578e1f8e65fb762cd26708c14cc6d" \
 --target "$DL/envs/lychee-vllm-deps-${SLURM_JOB_ID}" \
 --out "$DL/runs/lychee-vllm-deps-${SLURM_JOB_ID}"
