#!/bin/bash
#SBATCH --job-name=dl-dense1
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-dense1-%j.out

# PROBE 5 (dense-sh1): the DATA-side fix, owner-approved 2026-09-24.
# FIRST ATTEMPT (6843717) hit the 4h wall at ~step 240/400 — dense
# lessons are 2-3x longer than the sparse rig's (1400 vs 560-915
# ticks), ~50 s/step, no mid-training checkpoint survives. Retry with
# 8h walltime, same config.
# Probes 1-3 on the sparse rig mapped a knife edge (SH mutes / +ww3
# coexists at 37% / +pen re-mutes); root cause = ~95% sil ticks. The
# dense batch (synth-dense1) lifts word ticks from 7% to 12% overall
# and removes the 1-2% timed_task deserts; assistant gaps 2-5 s put
# small <t+Ns> tokens densely at correct onsets. ONE changed variable
# vs probe 2: the data. Recipe stays the probe-2 best (self-history
# 0.5, refresh 8, word-weight 3, NO penalty, timing-weight 50 default).
# Pass bar (free-run, trained 4): >=90% exact word-onset hits AND
# chatter <10% of silence ticks. FULL LOG — no grep pipes.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense1
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"

cd "$REPO"

echo "===== train: rows-only + SH 0.5 + ww3 on 4 DENSE conversations ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps 400 --lessons "$TRAIN4" --self-history 0.5 --sh-refresh 8 --word-weight 3 \
    --out $DL/runs/dense-sh1

for SET in "trained $TRAIN4" "heldout $HELD4"; do
  set -- $SET
  TAG=$1; IDS=$2
  echo "===== free-run dense-sh1 on $TAG dense conversations ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-sh1/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$IDS" --max-ticks 900
done
echo "dense probe done"
