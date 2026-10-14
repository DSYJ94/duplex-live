#!/bin/bash
# Finite native latency development check; no training or vendor changes.
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
SOURCE="$DL/personaplex"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 3428dfd95309a7f3c84fd93259ded0f810d1ff91
OVERLAY="$DL/tmp/voice-quality-7415052"
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$OVERLAY:$SOURCE/moshi:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo" \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" python -u -m evals.personaplex_recovery_probe \
  --batch "$DL/runs/synth-scale1-q/audio" --source "$SOURCE" \
  --model "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
  --voice "$DL/runs/voice-quality-7414595/codec/reference.wav" \
  --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json" \
  --out "$DL/runs/personaplex-paced-${SLURM_JOB_ID}" --paced
