#!/bin/bash
#SBATCH --job-name=dl-moshi-audit
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-moshi-audit-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
git rev-parse HEAD
apptainer exec --nv --bind /projects,/lus,/home --env PYTHONPATH="$DL/repo" \
    "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u -m evals.moshi_speech_audit \
    --runs "$DL/runs/moshi-base-7379833" "$DL/runs/moshi-adapter-7379833" \
    --out "$DL/runs/moshi-speech-audit-${SLURM_JOB_ID}"
