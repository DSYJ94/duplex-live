#!/bin/bash
#SBATCH --job-name=dl-daud4
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-daud4-%j.out

# AUDITION render for probe 8 (dense-pt2 = v2 + ww 3.0 — the first
# partially-SPEAKING v2 model): free-run the trained dense 4 with
# --save-picks (same command as the probe's scored free-run, so the
# audio matches the tables), then TTS the model's words onto the
# conversation audio -> runs/model-audio-dense-pt2 (gallery layout).
# Voices resolve per scenario from the LibriSpeech pool at
# $DL/models/voices. Expect sparse, misplaced speech (probe 8: speak
# acc 0.06-0.30, word acc ~0.05) — worth hearing what "dithering"
# sounds like by ear. FULL LOG — no grep pipes.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"

cd "$REPO"

echo "===== free-run dense-pt2 with --save-picks ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/dense-pt2/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$TRAIN4" --max-ticks 900 \
    --save-picks $DL/runs/picks-dense-pt2

echo "===== render dense-pt2 to listenable audio ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO \
    --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m evals.render_model_audio --picks $DL/runs/picks-dense-pt2 \
    --batch "$BATCH" --out $DL/runs/model-audio-dense-pt2
echo "dense-pt2 audition render done"
