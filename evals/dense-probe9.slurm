#!/bin/bash
#SBATCH --job-name=dl-dense5
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-dense5-%j.out

# PROBE 9 (dense-pt3): v2 labels + ww 3.0 + SH 0 — PLACEMENT probe 1
# (owner endpoint rule 2026-09-27: probe 8 spoke -> ladder shifts to
# placement). Same rig as probe 8 (dense-pt2: v2 shards-dense2,
# timing-weight 50, ww 3.0, trained 4 / held 4 dense lessons). ONE
# variable vs probe 8: --self-history 0.5 -> 0 (default). Why SH is
# the placement lever: the model never hears itself, so its own
# history is the only self-speech evidence in context — under SH 0.5
# half the steps condition on its own DITHERING (probe 8 recollects:
# 100% word babble at step 8 -> dithering -> late collapse). Teacher
# forcing gives complete, correct context while the placement mapping
# forms; exposure returns as a later curriculum step. Pass bar (v2):
# >=90% exact gold speech-run-onset hits AND false speech <10% of
# quiet ticks, trained 4. FULL LOG — no grep pipes.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"

cd "$REPO"

echo "===== train: rows-only + teacher forcing + ww3 on V2 shards ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps 400 --lessons "$TRAIN4" \
    --word-weight 3.0 \
    --out $DL/runs/dense-pt3

for SET in "trained $TRAIN4" "heldout $HELD4"; do
  set -- $SET
  TAG=$1; IDS=$2
  echo "===== free-run dense-pt3 on $TAG dense conversations ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$IDS" --max-ticks 900
done
echo "dense probe 9 done"
