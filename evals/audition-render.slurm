#!/bin/bash
#SBATCH --job-name=dl-audit2
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-audit2-%j.out

# AUDITION renders, take 2 (6827931 died inside MiniCPM-o TTS on one
# word; renderer now retries once and skips bad words). Picks for
# cap-rows / cap-sh / cap-sh2 already exist from 6827931 — render them
# AND add cap-sh3 (probe 3, re-muted) so the gallery tells the full
# four-checkpoint story: chatterer -> muted -> coexistence -> re-muted.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards
BATCH=$DL/runs/synth-big7/audio
TRAIN4="timed_task-001,timed_task-004,barge_in-000,barge_in-004"

cd "$REPO"

echo "===== free-run cap-sh3 with --save-picks (new arm) ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/cap-sh3/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$TRAIN4" --max-ticks 900 \
    --save-picks $DL/runs/picks-cap-sh3

for ARM in cap-rows cap-sh cap-sh2 cap-sh3; do
  echo "===== render $ARM to listenable audio ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO \
      --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m evals.render_model_audio --picks $DL/runs/picks-$ARM \
      --batch "$BATCH" --out $DL/runs/model-audio-$ARM
done
echo "audition renders done"
