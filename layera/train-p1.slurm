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

STEPS=${STEPS:-32}

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -u -m layera.train_p1 --shards $DL/runs/shards \
    --batch $DL/runs/synth-big7/audio --steps "$STEPS" --out $DL/runs/p1-v1

echo "[p1] checkpoint: $DL/runs/p1-v1"
