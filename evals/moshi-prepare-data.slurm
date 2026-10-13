#!/bin/bash
#SBATCH --job-name=dl-moshi-data
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-moshi-data-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
git rev-parse HEAD
ENV_DIR="$DL/envs/moshi-annotation-${SLURM_JOB_ID}"
ENV_OUT="$DL/runs/moshi-annotation-env-${SLURM_JOB_ID}"
OUT="$DL/runs/moshi-data-${SLURM_JOB_ID}"
export DL ENV_DIR ENV_OUT OUT
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH="$DL/repo" --env HF_HOME="$DL/models/hf-cache" \
    "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
python -m venv --system-site-packages "$ENV_DIR"
PY="$ENV_DIR/bin/python"
"$PY" -m evals.moshi_environment_setup --requirements "$DL/repo/evals/moshi-annotation-requirements.txt" --out "$ENV_OUT"
"$PY" -u -m evals.moshi_prepare_data \
    --batch "$DL/runs/synth-scale1-q/audio" --out "$OUT"
"$PY" -m pip freeze > "$ENV_OUT/pip-freeze.txt"
CONTAINER
