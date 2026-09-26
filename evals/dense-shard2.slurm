#!/bin/bash
#SBATCH --job-name=dl-shard2
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-shard2-%j.out

# Rebuild the dense shards under the V2 label contract (per-tick
# speech, ADR-0004 2026-09-26): the grid now emits <speak> on every
# utterance-internal continuation tick, so speech labels are 59.7% of
# ticks vs sil 39.9% (v1: 12%/88%). Same 12 filter-passing lessons as
# shards-dense1 (the manifest already holds exactly those), same audio
# batch — ONLY the labels change. shards-dense1 stays untouched for
# v1 comparisons.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

cd "$REPO"

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.prepare --batch $DL/runs/synth-dense1/audio \
        --out $DL/runs/shards-dense2 --per-class 8
echo "v2 shards built"
