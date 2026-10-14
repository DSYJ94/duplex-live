#!/bin/bash
# A single short captured-inference reference diagnostic.
set -euo pipefail
DL=/projects/u6si/duplex-live
SOURCE="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
export DL SOURCE
cd "$DL/repo"
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$SOURCE/moshi:$DL/repo" --env HF_HOME="$DL/models/hf-cache" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=8 "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
PY="$DL/envs/moshi-rag-7379661/bin/python"
export TRITON_LIBCUDA_PATH
TRITON_LIBCUDA_PATH=$("$PY" -m evals.cuda_linker_path --out "$DL/runs/reference-effect-driver-${SLURM_JOB_ID}")
"$PY" -u -m evals.moshi_reference_effect --cache "$DL/runs/recovery-codes-7408223" \
  --manifest "$DL/runs/moshi-models-7378305/model-manifest.json" \
  --out "$DL/runs/reference-effect-${SLURM_JOB_ID}"
CONTAINER
