#!/bin/bash
#SBATCH --job-name=freerun
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/freerun-%j.out
#
# Free-run diagnostic: teacher-forced vs self-fed history on held-out
# factory lessons (exposure-bias check), + ear token-rate alignment.
# See layera/freerun_diag.py.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

CLASSES=${CLASSES:-timed_task}
LIMIT=${LIMIT:-2}
CKPT=${CKPT:-$DL/runs/p1-v3/p1_checkpoint.pt}
SHARDS=${SHARDS:-$DL/runs/shards}

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -u -m layera.freerun_diag --batch $DL/runs/synth-big7/audio \
    --ckpt "$CKPT" --shards "$SHARDS" --classes "$CLASSES" --limit "$LIMIT"

echo "[freerun] all done"
