#!/bin/bash
#SBATCH --job-name=dl-m2par1
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-m2par1-%j.out

# M2 milestone 1 (owner strategy 2026-09-27: bounded ladder + M2 in
# parallel): real-time 80 ms tick PARITY — the docs/04 shadow test.
# Runs the exact offline free-run decode (a) as fast as possible and
# (b) paced to the true tick clock on the 4 trained dense lessons,
# then gates: schedule adherence >= 99.9% (docs/04 SLO), 0 per-tick
# emission mismatches (the clock must not change what the model says),
# p999 decision latency < 80 ms. Checkpoint = dense-pt1 (probe 7):
# the loop mechanics don't need a good model, just A model; rerun on
# dense-pt2 when probe 8 lands. FULL LOG — no grep pipes.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"

cd "$REPO"

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m m2.parity --ckpt $DL/runs/dense-pt1/p1_checkpoint.pt \
    --shards "$SHARDS" --scenarios "$TRAIN4" \
    --out $DL/runs/m2-parity-dense-pt1
echo "m2 parity done"
