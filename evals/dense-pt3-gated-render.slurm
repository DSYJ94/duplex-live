#!/bin/bash
#SBATCH --job-name=dl-dgate2
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-dgate2-%j.out

# RENDER-ONLY fix for dl-dgate (6911470): the gated free-run finished
# and --save-picks wrote runs/picks-dense-pt3-gated, but the TTS stage
# died on `ImportError: minicpmo` — the job's PYTHONPATH lacked the
# TTS overlay. First re-submit (6911492) still failed because I wrote
# the overlay name wrong: it is pylibs-minicpm (no "o"), as in
# dense-audition5/6. Render the existing picks with the right env.
# FULL LOG — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO

BATCH=$DL/runs/synth-dense1/audio

cd "$REPO"

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m evals.render_model_audio --picks $DL/runs/picks-dense-pt3-gated \
    --batch "$BATCH" --out $DL/runs/model-audio-dense-pt3-gated
echo "gated audition render done"
