#!/bin/bash
#SBATCH --job-name=dl-recovery-train
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-recovery-train-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
SOURCE="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
cd "$DL/repo"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 8c6dfc101b7871baa428424bcdc583b74fb561d9
export DL SOURCE
export RECOVERY_CACHE="${RECOVERY_CACHE:?Verified encoded cache required}"
export RECOVERY_STEPS="${RECOVERY_STEPS:-32}"
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$SOURCE/moshi:$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
PY="$DL/envs/moshi-rag-7379661/bin/python"
export TRITON_LIBCUDA_PATH
TRITON_LIBCUDA_PATH=$("$PY" -m evals.cuda_linker_path --out "$DL/runs/recovery-train-driver-${SLURM_JOB_ID}")
ARGS=(--cache "$RECOVERY_CACHE" --manifest "$DL/runs/moshi-models-7378305/model-manifest.json" \
  --out "$DL/runs/recovery-training-${SLURM_JOB_ID}" --steps "$RECOVERY_STEPS")
"$PY" -u -m evals.recovery_train "${ARGS[@]}"
"$PY" -u -m evals.recovery_train "${ARGS[@]}" --reload
CONTAINER
