#!/bin/bash
#SBATCH --job-name=dl-sq2demo
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/dl-sq2demo-%j.out

# HELD-OUT CONFIRMATION + AUDIBLE DEMO (2026-10-04). Operating point
# picked on TRAIN4 (thr=0.10 — the only one where every lesson works;
# higher thresholds mute dialog-010). Protocol: ONE held-out shot at
# the picked point, picks saved, then the picks are rendered to audio
# with the factory TTS so the standing listen rule applies
# (viz.listen). Held-out lessons only — the honest demo.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

BATCH=$DL/runs/synth-scale1-q/audio
SHARDS=$DL/runs/shards-scale1-q2
CKPT=$DL/runs/scale1-q2-pt1/p1_checkpoint.pt
GATE=$DL/runs/scale1-q2-gate1/gate_checkpoint.pt
HELD="dense_dialog-114,dense_dialog-116,dense_barge-113,dense_barge-114"
OUT=$DL/runs/demo-q2

cd "$REPO"
mkdir -p "$OUT"

echo "===== held-out confirmation: gate thr=0.10, picks saved ====="
export PYTHONPATH=$DL/pylibs:$REPO
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER \
    python -m layera.freerun_diag --batch "$BATCH" --ckpt "$CKPT" \
    --shards "$SHARDS" --scenarios "$HELD" \
    --gate-ckpt "$GATE" --gate-threshold 0.10 \
    --save-picks "$OUT/picks"

echo "===== render picks to audio (factory TTS, judge on) ====="
export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO
ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
VARGS=""
[ -f "$VJ" ] && VARGS="--voices-pool $VJ"
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER \
    python -m evals.render_model_audio --picks "$OUT/picks" \
    --batch "$BATCH" --out "$OUT/audio" --engine minicpm $VARGS \
   

echo "demo audio done: $OUT/audio — rsync + viz.listen per the standing rule"
