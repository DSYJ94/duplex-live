#!/bin/bash
#SBATCH --job-name=dl-gate1
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-gate1-%j.out

# AUX ACTIVITY GATE (owner decision 2026-09-28, after probe 11): the
# three-point law says one head carrying placement + discipline trades
# them ~1:1 under any history corruption. So freeze dense-pt3
# (placement, untouchable) and train a SEPARATE small MLP on the same
# per-tick trunk hidden state that feeds the word head: binary
# "should anyone be speaking this tick?", plain balanced CE, half the
# samples on corrupt-gold histories (probe-11 corruption verbatim) so
# the gate sees the off-gold states decode visits. Decode: a word
# counts only if the model proposes it AND the gate says speech.
# Sweep thresholds 0.35/0.5/0.65 trained + held 0.5. Pass bar (v2):
# >=90% exact run-onsets AND false speech <10% of quiet ticks. FULL
# LOG — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"

cd "$REPO"

echo "===== train activity gate on FROZEN dense-pt3 (200 steps) ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_gate --shards "$SHARDS" --batch "$BATCH" \
    --steps 200 --lessons "$TRAIN4" --corrupt-prob 0.5 \
    --init-ckpt $DL/runs/dense-pt3/p1_checkpoint.pt \
    --out $DL/runs/dense-gate1

for TH in 0.35 0.5 0.65; do
  echo "===== free-run pt3+gate thr=$TH on trained lessons ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$TRAIN4" --max-ticks 900 \
      --gate-ckpt $DL/runs/dense-gate1/gate_checkpoint.pt \
      --gate-threshold $TH
done

echo "===== free-run pt3+gate thr=0.5 on heldout lessons ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$HELD4" --max-ticks 900 \
    --gate-ckpt $DL/runs/dense-gate1/gate_checkpoint.pt \
    --gate-threshold 0.5
echo "dense gate 1 done"
