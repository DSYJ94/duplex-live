#!/bin/bash
#SBATCH --job-name=dl-dense2
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-dense2-%j.out

# PROBE 6 (dense-sh2): owner-approved 2026-09-25, the pre-registered
# next dose after probe 5. ONE changed variable vs probe 5: word-weight
# 3 -> 6 (dense grid is 12% word ticks, so effective word share goes
# ~36% -> ~72%). Probe 5 verdict: UNMUTED (chatter 7.5%, under the 10%
# bar) but word placement uncorrelated with gold (exact 5.6%, ±1s hits
# at chance). This dose should fix the under-speaking (5.4% word ticks
# vs 12% gold); whether it sharpens PLACEMENT is the real test — if
# not, the labeling contract (one word tick per spoken word) is the
# last suspect. Same pass bar (>=90% exact onsets, <10% chatter on
# trained 4). 8h walltime (dense lessons ~2-3x longer than sparse).
# FULL LOG — no grep pipes.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense1
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"

cd "$REPO"

echo "===== train: rows-only + SH 0.5 + ww6 on 4 DENSE conversations ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps 400 --lessons "$TRAIN4" --self-history 0.5 --sh-refresh 8 --word-weight 6 \
    --out $DL/runs/dense-sh2

for SET in "trained $TRAIN4" "heldout $HELD4"; do
  set -- $SET
  TAG=$1; IDS=$2
  echo "===== free-run dense-sh2 on $TAG dense conversations ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-sh2/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$IDS" --max-ticks 900
done
echo "dense probe 6 done"
