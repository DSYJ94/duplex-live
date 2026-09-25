#!/bin/bash
#SBATCH --job-name=dl-daud1
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-daud1-%j.out

# AUDITION render for probe 5 (dense-sh1): free-run the trained dense
# 4 with --save-picks (same command as the probe's scored free-run, so
# the audio matches the tables), then TTS the model's words onto the
# conversation audio -> runs/model-audio-dense-sh1 (gallery layout).
# Owner tracks every probe by ear; renderer already retries+skips
# MiniCPM-o's occasional mid-word crash.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense1
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"

cd "$REPO"

echo "===== free-run dense-sh1 with --save-picks ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/dense-sh1/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$TRAIN4" --max-ticks 900 \
    --save-picks $DL/runs/picks-dense-sh1

echo "===== render dense-sh1 to listenable audio ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO \
    --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m evals.render_model_audio --picks $DL/runs/picks-dense-sh1 \
    --batch "$BATCH" --out $DL/runs/model-audio-dense-sh1
echo "dense audition render done"
