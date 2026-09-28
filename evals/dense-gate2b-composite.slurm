#!/bin/bash
#SBATCH --job-name=dl-gate2b
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/dl-gate2b-%j.out

# GATE2 COMPOSITE PROBE (2026-09-28, after the gate2 verdict): gate2
# fixed onset ranking (16/16 exact at thr 0.5 on TRAIN4) but chatter
# regressed (31-77% of quiet ticks — the base model proposes a word on
# ~91% of quiet ticks and gate2 accepts ~84% of those proposals). The
# min-gap rule blocks word-first picks for `gap` ticks after every
# ACCEPTED word, so it divides chatter by ~gap; gold speech-RUN gaps
# are >= 93 ticks everywhere, so gaps 8-16 cannot clip a real response
# start. The one risk: a chatter accept sitting < gap ticks before a
# gold onset pushes that onset late (exact -> near). This probe measures
# exactly that trade. Arms: TRAIN4 gap 8 (save-picks, render candidate),
# TRAIN4 gap 16, HELD4 gap 8; all thr 0.5 + user-gate hold 3 (hold 3 is
# onset-safe: tightest gold user->assistant gap is 5 ticks). Pass bar
# v2: >= 90% exact onsets AND false speech < 10% of quiet ticks.
# FULL LOG — no grep at run time.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"
GATE=$DL/runs/dense-gate2/gate_checkpoint.pt

cd "$REPO"

echo "===== gate2 composite thr=0.5 gap=8 + user-gate hold 3, TRAIN4 ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$TRAIN4" --max-ticks 900 \
    --gate-ckpt "$GATE" --gate-threshold 0.5 \
    --min-gap-ticks 8 --user-gate --gate-hold 3 \
    --save-picks $DL/runs/picks-gate2-composite-gap8

echo "===== gate2 composite thr=0.5 gap=16 + user-gate hold 3, TRAIN4 ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$TRAIN4" --max-ticks 900 \
    --gate-ckpt "$GATE" --gate-threshold 0.5 \
    --min-gap-ticks 16 --user-gate --gate-hold 3

echo "===== gate2 composite thr=0.5 gap=8 + user-gate hold 3, HELD4 ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$HELD4" --max-ticks 900 \
    --gate-ckpt "$GATE" --gate-threshold 0.5 \
    --min-gap-ticks 8 --user-gate --gate-hold 3

echo "dense gate 2b composite done"
