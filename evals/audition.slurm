#!/bin/bash
#SBATCH --job-name=dl-audition
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-audition-%j.out

# AUDITION renders (owner request 2026-09-23: "let me listen via the
# localhost gallery to track progress"). Free-runs the 4 trained
# conversations under all three self-history-series checkpoints with
# --save-picks, then speaks each model's words (factory TTS voice) at
# the exact ticks the model chose, over the conversation audio. Output
# layout feeds viz.listen directly; rsync runs/model-audio-* to the
# local Mac and open the gallery.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards
BATCH=$DL/runs/synth-big7/audio
TRAIN4="timed_task-001,timed_task-004,barge_in-000,barge_in-004"

cd "$REPO"

for ARM in cap-rows cap-sh cap-sh2; do
  echo "===== free-run $ARM with --save-picks ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/$ARM/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$TRAIN4" --max-ticks 900 \
      --save-picks $DL/runs/picks-$ARM
done

for ARM in cap-rows cap-sh cap-sh2; do
  echo "===== render $ARM to listenable audio ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO \
      --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m evals.render_model_audio --picks $DL/runs/picks-$ARM \
      --batch "$BATCH" --out $DL/runs/model-audio-$ARM
done
echo "audition renders done"
