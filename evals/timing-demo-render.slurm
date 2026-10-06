#!/bin/bash
#SBATCH --job-name=dl-timingdemo
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-timingdemo-%j.out

# TIMING-DEMO RENDER (2026-10-05): the honest audible demo — the
# model's turn onsets + the script's words (free-run content is digit
# soup; placement is real). Picks built by evals.build_timing_demo
# (job 7113366 step 1: barge-113 3/3, barge-114 2/4, dialog-114 2/7,
# dialog-116 7/7 turns started). This job renders them with the
# factory TTS + judge. Output: $DL/runs/demo-q2/audio-timing.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

cd "$REPO"

export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO
ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
VARGS=""
[ -f "$VJ" ] && VARGS="--voices-pool $VJ"

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER \
    python -m evals.render_model_audio \
    --picks $DL/runs/demo-q2/picks-timing \
    --batch $DL/runs/synth-scale1-q/audio \
    --out $DL/runs/demo-q2/audio-timing \
    --engine minicpm $VARGS

echo "timing demo done: $DL/runs/demo-q2/audio-timing"
