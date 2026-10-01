#!/bin/bash
#SBATCH --job-name=dl-sc1hd1
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-sc1hd1-%j.out

# RL-PROBE 1: head-DAgger on the scaled trunk (2026-10-01, per the
# owner-approved two-probe RL ladder). Gate1-on-scale1 verdict: a side
# gate on the scaled states FAILS (0/17 held exact) — the five-point
# law extends to 33x states. This probe retrains the model's OWN
# duplex head rows on DAgger rollout states (raw decode, gold tokens):
# the word/quiet competition is edited at the source instead of gated
# beside it. Trunk + word rows + input-embedding deltas stay frozen.
# Stages: rollout (8 lessons, ~1.5 h) + head train (~minutes) -> save
# runs/dagger-states/scale1.pt for cheap re-iteration -> UNGATED
# freerun on TRAIN4 and on the SAME 4 held lessons as the scale1
# closing freerun (direct comparison with the ungated verdict table).
# Pass bar v2: >=90% exact onsets AND false speech <10% of quiet ticks.
# FULL LOG — no grep at run time.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-scale1
BATCH=$DL/runs/synth-scale1/audio
BASE=$DL/runs/scale1-pt1/p1_checkpoint.pt
OUT=$DL/runs/scale1-headdag

ROLL="dense_dialog-001,dense_dialog-010,dense_dialog-012,dense_dialog-020,dense_barge-001,dense_barge-004,dense_barge-007,dense_barge-010"
TRAIN4="dense_dialog-001,dense_dialog-010,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-116,dense_dialog-118,dense_barge-113,dense_barge-114"

cd "$REPO"

echo "===== head-DAgger: duplex-row train (RESUME: saved rollout states) ====="
# job 7001927 crashed at train start (fp32 states @ bf16 W, fixed in
# train_head_dagger.py) AFTER saving the 4,374-state rollout buffer —
# --states skips the ~1.5h rollout entirely (the designed resume path).
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_head_dagger --shards "$SHARDS" --batch "$BATCH" \
    --init-ckpt "$BASE" --out "$OUT" \
    --lessons "$ROLL" \
    --states $DL/runs/dagger-states/scale1.pt \
    --save-states $DL/runs/dagger-states/scale1.pt \
    --steps 300 --lr 1e-3 --buffer-batch 512

echo "===== free-run headdag UNGATED on trained lessons ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt "$OUT/p1_checkpoint.pt" --shards "$SHARDS" \
    --scenarios "$TRAIN4" --max-ticks 900

echo "===== free-run headdag UNGATED on the scale1 held-out four ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt "$OUT/p1_checkpoint.pt" --shards "$SHARDS" \
    --scenarios "$HELD4" --max-ticks 900
echo "scale1 headdag done"
