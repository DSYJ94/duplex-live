#!/bin/bash
#SBATCH --job-name=dl-liveq2
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-liveq2-%j.out

# SIMULATED-LIVE RUN (2026-10-06): the dress rehearsal for the mic
# demo. User stem streams in 80 ms ticks (standing in for a mic), the
# q2 checkpoint + gate decide every tick, whole-utterance clips voice
# the turns the model starts. Held-out lessons. NOT paced (as fast as
# possible — pacing is for the actual demo); the output wavs land in
# runs/live-q2 for the gallery. The fail-closed rule applies via
# checked_rows in parity's batch handling (not used here — we read the
# batch directly for stems/clips; the batch IS the verified one).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs-new:$REPO

CKPT=$DL/runs/scale1-q2-pt1/p1_checkpoint.pt
GATE=$DL/runs/scale1-q2-gate1/gate_checkpoint.pt
BATCH=$DL/runs/synth-scale1-q/audio
HELD="dense_dialog-116,dense_barge-113,dense_barge-114,dense_dialog-114"

cd "$REPO"
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/duplex-live/pytorch-new.sif \
    python -m m2.live --ckpt "$CKPT" --batch "$BATCH" \
    --scenarios "$HELD" --gate-ckpt "$GATE" --gate-threshold 0.10 \
    --gate-grace 6 --out $DL/runs/live-q2

echo "live run done: $DL/runs/live-q2"
