#!/bin/bash
#SBATCH --job-name=p1-train
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/p1-train-%j.out
#
# Phase-1 v1: first gradients on factory data. Frozen brain, only the new
# duplex tokens train (row-masked so old vocabulary cannot drift).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STEPS=${STEPS:-32}
OUT=${OUT:-$DL/runs/p1-v1}
SHARDS=${SHARDS:-$DL/runs/shards}
TIMING_WEIGHT=${TIMING_WEIGHT:-50.0}
EXTRA=()
[ -n "${TIMING_WEIGHT:-}" ] && EXTRA+=(--timing-weight "$TIMING_WEIGHT")
[ -n "${WEIGHT_YIELD:-}" ] && EXTRA+=(--weight-yield)
[ -n "${CLASS_COST:-}" ] && EXTRA+=(--class-cost "$CLASS_COST")
[ -n "${THIEF_PENALTY:-}" ] && EXTRA+=(--thief-penalty "$THIEF_PENALTY")
[ -n "${WORD_WEIGHT:-}" ] && EXTRA+=(--word-weight "$WORD_WEIGHT")

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -u -m layera.train_p1 --shards "$SHARDS" \
    --batch $DL/runs/synth-big7/audio --steps "$STEPS" --out "$OUT" \
    "${EXTRA[@]}"

echo "[p1] checkpoint: $OUT"
