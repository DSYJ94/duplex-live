#!/bin/bash
# Payload for a bounded interactive reload diagnostic, not a new training run.
set -euo pipefail
: "${RAG_RELOAD_RUN:?Set an existing optimizer pilot directory}"
DL=/projects/u6si/duplex-live
cd "$DL/repo"
SOURCE="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$SOURCE/moshi:$DL/repo" --env HF_HOME="$DL/models/hf-cache" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" -u \
  -m evals.moshi_rag_training_probe --reload \
  --manifest "$DL/runs/moshi-models-7378305/model-manifest.json" \
  --data "$DL/runs/moshi-data-7379026" \
  --native-run "$DL/runs/moshi-training-7380505" --out "$RAG_RELOAD_RUN"
