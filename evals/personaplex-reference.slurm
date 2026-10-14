#!/bin/bash
# Finite engineering diagnostic, no optimizer, no semantic qualification.
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
SOURCE="$DL/personaplex"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 3428dfd95309a7f3c84fd93259ded0f810d1ff91
export CUBLAS_WORKSPACE_CONFIG=:4096:8
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$SOURCE/moshi:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo" \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env HF_HUB_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" python -u -m evals.personaplex_reference_probe \
  --cache "$DL/runs/recovery-codes-7408223" \
  --model "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
  --out "$DL/runs/personaplex-reference-${SLURM_JOB_ID}"
