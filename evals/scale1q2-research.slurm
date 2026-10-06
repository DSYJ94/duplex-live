#!/bin/bash
#SBATCH --job-name=dl-sq2res
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-sq2res-%j.out

# RESEARCH RUN on the EAR-ALIGNED trunk (owner directive 2026-10-06:
# "as close to GPT-live as possible; full permission; set a goal").
# GOAL: free-run WORD WER <= 0.5 on held-out by Oct 20 (today ~1.0,
# digit soup) while keeping placement. Premise to re-test: D1's "no
# turn signal in trunk states" was measured on the SKEWED-ear trunk —
# with hearing fixed the readout path (and with it real word
# generation) may reopen. This job:
#   1. fresh DAgger rollouts on q2 -> saved states
#   2. D1 probes ON THE Q2 STATES (the decisive rerun)
#   3. DAgger head trained on those states
#   4. free-run content baseline (FREE-RUN WORD WER) for q2 and the
#      dagger head — the before number for the SH round.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif
APPT="apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER"

SHARDS=$DL/runs/shards-scale1-q2
BATCH=$DL/runs/synth-scale1-q/audio
BASE=$DL/runs/scale1-q2-pt1/p1_checkpoint.pt
OUT=$DL/runs/scale1-q2-headdag
ROLL="dense_dialog-001,dense_dialog-010,dense_dialog-012,dense_dialog-020,dense_barge-001,dense_barge-004,dense_barge-007,dense_barge-010"
TRAIN4="dense_dialog-001,dense_dialog-010,dense_barge-001,dense_barge-004"

cd "$REPO"

echo "===== 1/4 head-DAgger rollouts on q2 (fresh states) ====="
$APPT python -m layera.train_head_dagger --shards "$SHARDS" --batch "$BATCH" \
    --init-ckpt "$BASE" --out "$OUT" \
    --lessons "$ROLL" \
    --save-states $DL/runs/dagger-states/q2.pt \
    --steps 300 --lr 1e-3 --buffer-batch 512

echo "===== 2/4 D1 probes on the Q2 STATES (the decisive rerun) ====="
$APPT python -m layera.diag_states --states $DL/runs/dagger-states/q2.pt

echo "===== 3/4 free-run content baseline: q2 trunk ====="
$APPT python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt "$BASE" --shards "$SHARDS" --scenarios "$TRAIN4"

echo "===== 4/4 free-run: q2 + dagger head ====="
$APPT python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt "$OUT/p1_checkpoint.pt" --shards "$SHARDS" --scenarios "$TRAIN4"

echo "research run done"
