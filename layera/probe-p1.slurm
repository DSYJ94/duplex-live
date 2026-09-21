#!/bin/bash
#SBATCH --job-name=p1-probe
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/p1-probe-%j.out
#
# Phase-1 probe: held-out scenarios, before-vs-after checkpoint, per
# token type. See layera/probe_p1.py.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -u -m layera.probe_p1 --batch $DL/runs/synth-big7/audio \
    --ckpt $DL/runs/p1-v1/p1_checkpoint.pt --skip 8 --limit 4

echo "[probe] done"
