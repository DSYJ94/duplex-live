#!/bin/bash
#SBATCH --job-name=dl-sq2gsweep
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-sq2gsweep-%j.out

# GATE THRESHOLD SWEEP on TRAINING lessons (2026-10-04). The q2 gate at
# thr=0.5 over-blocks on unseen lessons (recall 0.03 on dialog-114) —
# its onset probability is calibrated lower out-of-distribution. Pick
# the operating point on TRAIN4 (training lessons, already burned for
# tuning — NOT held-out), then ONE held-out shot at the winner. Both
# sil-bias 0 (pure gate) and the raw ungated baseline for reference.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs-new:$REPO
CONTAINER=/projects/u6si/duplex-live/pytorch-new.sif

BATCH=$DL/runs/synth-scale1-q/audio
SHARDS=$DL/runs/shards-scale1-q2
CKPT=$DL/runs/scale1-q2-pt1/p1_checkpoint.pt
GATE=$DL/runs/scale1-q2-gate1/gate_checkpoint.pt
SWEEP="dense_dialog-001,dense_dialog-010,dense_barge-001,dense_barge-004"

cd "$REPO"
for THR in 0.10 0.20 0.35 0.50; do
  echo "===== gate thr=$THR (TRAIN4) ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      $CONTAINER \
      python -m layera.freerun_diag --batch "$BATCH" --ckpt "$CKPT" \
      --shards "$SHARDS" --scenarios "$SWEEP" \
      --gate-ckpt "$GATE" --gate-threshold "$THR" \
      || echo "[sweep] thr=$THR run failed (output above is the result)"
done
echo "===== ungated reference (TRAIN4) ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER \
    python -m layera.freerun_diag --batch "$BATCH" --ckpt "$CKPT" \
    --shards "$SHARDS" --scenarios "$SWEEP"
echo "gate sweep done"
