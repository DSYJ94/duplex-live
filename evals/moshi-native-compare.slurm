#!/bin/bash
#SBATCH --job-name=dl-moshi-voices
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-moshi-voices-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
NATIVE="$DL/vendor/moshi-e6a55d2722a65870ef52a6c9f6ecfc0e90f38362"
ENV_DIR="$DL/envs/moshi-native-7379340"
TRAIN="$DL/runs/moshi-training-7379340"
git rev-parse HEAD
export DL ENV_DIR TRAIN
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH="$NATIVE/moshi:$DL/repo" \
    --env TRITON_LIBCUDA_PATH="$DL/runs/moshi-training-env-7379340/driver" \
    --env TMPDIR="$DL/tmp/moshi-native-build-7379340" "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
PY="$ENV_DIR/bin/python"
COMMON=(--model-manifest "$DL/runs/moshi-models-7378305/model-manifest.json"
        --batch "$DL/runs/synth-scale1-q/audio" --scenario dense_dialog-102 --seconds 40)
"$PY" -u -m evals.moshi_native_replay "${COMMON[@]}" --config "$TRAIN/base-config.json" \
    --out "$DL/runs/moshi-base-${SLURM_JOB_ID}"
"$PY" -u -m evals.moshi_native_replay "${COMMON[@]}" \
    --config "$TRAIN/train/checkpoints/checkpoint_000004/consolidated/config.json" \
    --adapter "$TRAIN/train/checkpoints/checkpoint_000004/consolidated/lora.safetensors" \
    --out "$DL/runs/moshi-adapter-${SLURM_JOB_ID}"
CONTAINER
