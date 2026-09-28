#!/bin/bash
#SBATCH --job-name=dl-gate2
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/dl-gate2-%j.out

# GATE ROUND 2 (owner decision 2026-09-28, after gate1 half-worked):
# DAgger-style — collect the states decode ACTUALLY visits by free-
# running each TRAIN4 lesson with gate1 ACTIVE at threshold 0.35 (the
# exact decode config of gate1's sweep), record the hidden state at
# every word-first proposal tick with its GOLD label, then train the
# gate on that buffer mixed with the gate1 gold+corrupt stream.
# Hypothesis under test: gate1 failed because its training states
# missed decode's; if round-2 scores separate (gold onsets ranked
# above chatter), Layer A has its recipe. Sweep 0.35/0.5/0.65 trained
# + held 0.5, identical to gate1 for direct comparison. freerun prints
# the gate's P at gold speech onsets per lesson. FULL LOG — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"

cd "$REPO"

echo "===== dagger rollout + gate round 2 on FROZEN dense-pt3 ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_gate --shards "$SHARDS" --batch "$BATCH" \
    --steps 200 --lessons "$TRAIN4" --corrupt-prob 0.5 --dagger \
    --dagger-gate $DL/runs/dense-gate1/gate_checkpoint.pt \
    --rollout-threshold 0.35 \
    --init-ckpt $DL/runs/dense-pt3/p1_checkpoint.pt \
    --out $DL/runs/dense-gate2

for TH in 0.35 0.5 0.65; do
  echo "===== free-run pt3+gate2 thr=$TH on trained lessons ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$TRAIN4" --max-ticks 900 \
      --gate-ckpt $DL/runs/dense-gate2/gate_checkpoint.pt \
      --gate-threshold $TH
done

echo "===== free-run pt3+gate2 thr=0.5 on heldout lessons ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$HELD4" --max-ticks 900 \
    --gate-ckpt $DL/runs/dense-gate2/gate_checkpoint.pt \
    --gate-threshold 0.5
echo "dense gate 2 done"
