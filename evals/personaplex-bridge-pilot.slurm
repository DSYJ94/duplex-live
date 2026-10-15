#!/bin/bash
# Eight-step native engineering pilot; one TRAIN sample, frozen speech model.
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
SOURCE="$DL/personaplex"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 3428dfd95309a7f3c84fd93259ded0f810d1ff91
ARGS=(--cache "$DL/runs/recovery-codes-7408223"
  --model "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
  --voice "$DL/runs/voice-quality-7414595/codec/reference.wav"
  --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json"
  --out "$DL/runs/personaplex-bridge-${SLURM_JOB_ID}")
run_python() {
  apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$SOURCE/moshi:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo" \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env HF_HUB_OFFLINE=1 \
    --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
    "$DL/pytorch-new.sif" python -u -m evals.personaplex_bridge_pilot "${ARGS[@]}" "$@"
}
run_python
run_python --reload
