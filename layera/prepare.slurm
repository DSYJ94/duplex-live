#!/bin/bash
#SBATCH --job-name=layera-prep
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/layera-prep-%j.out
#
# Phase-1 prep: encode factory scenarios into training shards
# (in-slot audio -> AuT features; grid emissions -> answer key).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

BATCH=${BATCH:-$DL/runs/synth-big7/audio}
OUT=${OUT:-$DL/runs/shards}
LIMIT=${LIMIT:-8}
PER_CLASS=${PER_CLASS:-0}

EXTRA=()
if [ "$PER_CLASS" -gt 0 ]; then
    EXTRA+=(--per-class "$PER_CLASS")
fi

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -u -m layera.prepare --batch "$BATCH" --out "$OUT" --limit "$LIMIT" \
    "${EXTRA[@]}"

echo "[prep] shards: $OUT"
