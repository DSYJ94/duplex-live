#!/bin/bash
#SBATCH --job-name=dl-moshi-reload
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-moshi-reload-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
NATIVE="$DL/vendor/moshi-e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
ENV_DIR="$DL/envs/moshi-native-7379340"
git rev-parse HEAD
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH="$NATIVE/moshi:$DL/repo" \
    --env TRITON_LIBCUDA_PATH="$DL/runs/moshi-training-env-7379340/driver" \
    --env TMPDIR="$DL/tmp/moshi-native-build-7379340" "$DL/pytorch-new.sif" \
    "$ENV_DIR/bin/python" -u -m evals.moshi_reload_diagnostic \
    --run "$DL/runs/moshi-training-7379340" \
    --model-manifest "$DL/runs/moshi-models-7378305/model-manifest.json" \
    --out "$DL/runs/moshi-reload-${SLURM_JOB_ID}"
