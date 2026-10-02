#!/bin/bash
#SBATCH --job-name=dl-sc1ug
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-sc1ug-%j.out

# THE DIRECT PATH (2026-10-02, per the owner's "target the problem
# directly"): scale1-pt1 has SOLVED placement (probe-9 recipe at 33x);
# the missing ingredient per the capacity test is speech/silence
# GATING — and the only untapped INDEPENDENT channel is mic energy
# (the model's own voice is never in its ears, so a mic-side VAD sees
# exactly what this mask sees). This job composes the two on the held
# lessons: --user-gate (energy blocks word-FIRST picks; words in
# flight complete) + a gate-hold sweep. NO min-gap: the demo config
# dropped it ("strangled responses", HANDOFF 2026-09-30 composite
# verdict) and m2 parity never had it — policy parity with the demo
# config matters more than an extra knob. No training — existing
# freerun_diag decode rules only. Every config saves picks for audio
# rendering + viz.listen (the ear is the verdict).
# Panel vs the scale1 baseline decides. FULL LOG — no grep at run time.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$REPO

SHARDS=$DL/runs/shards-scale1
BATCH=$DL/runs/synth-scale1/audio
BASE=$DL/runs/scale1-pt1/p1_checkpoint.pt
PICKS=$DL/runs/scale1-ugate/picks

HELD4="dense_dialog-116,dense_dialog-118,dense_barge-113,dense_barge-114"
TRAIN4="dense_dialog-001,dense_dialog-010,dense_barge-001,dense_barge-004"

cd "$REPO"

for HOLD in 5 12; do
  echo "===== HELD4 user-gate hold=$HOLD (no min-gap) ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt "$BASE" --shards "$SHARDS" \
      --scenarios "$HELD4" --max-ticks 900 \
      --user-gate --gate-hold "$HOLD" \
      --save-picks "$PICKS-hold$HOLD"
done

echo "===== TRAIN4 user-gate hold=5 (no min-gap) ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt "$BASE" --shards "$SHARDS" \
    --scenarios "$TRAIN4" --max-ticks 900 \
    --user-gate --gate-hold 5 \
    --save-picks "$PICKS-train4-hold5"
echo "scale1 ugate done"
