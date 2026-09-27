#!/bin/bash
#SBATCH --job-name=dl-daud5
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-daud5-%j.out

# AUDITION render for probe 9 (dense-pt3 = v2 + ww3 + teacher forcing
# — perfect placement, constant chatter): free-run the trained dense 4
# with --save-picks, then TTS -> runs/model-audio-dense-pt3 (gallery).
# Expect a model that interjects words in every gap — the audible face
# of precision 0.11-0.14. Pairs with model-audio-dense-pt2 (probe 8's
# dithering) on the gallery. FULL LOG — no grep pipes.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"

cd "$REPO"

echo "===== free-run dense-pt3 with --save-picks ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$TRAIN4" --max-ticks 900 \
    --save-picks $DL/runs/picks-dense-pt3

echo "===== render dense-pt3 to listenable audio ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO \
    --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m evals.render_model_audio --picks $DL/runs/picks-dense-pt3 \
    --batch "$BATCH" --out $DL/runs/model-audio-dense-pt3
echo "dense-pt3 audition render done"
