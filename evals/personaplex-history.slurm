#!/bin/bash
# Finite native delayed-history parity check, no optimization.
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/personaplex/moshi:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo" \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" python -u -m evals.personaplex_history_probe \
  --cache "$DL/runs/recovery-codes-7408223" \
  --model "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
  --voice "$DL/runs/voice-quality-7414595/codec/reference.wav" \
  --template "$DL/runs/personaplex-bridge-7428894" \
  --out "$DL/runs/personaplex-history-${SLURM_JOB_ID}"
