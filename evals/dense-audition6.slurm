#!/bin/bash
#SBATCH --job-name=dl-daud6
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-daud6-%j.out

# RE-RENDER of dense-pt3 with the fixed mixer (owner report
# 2026-09-27: "groaning deep noise, model speech soft in the
# background"). Root cause: probe 9's chatter stacks up to ~8 word
# clips per instant; the summed peak then triggered whole-mix
# normalization, crushing everything by the worst pileup's factor.
# The mixer now soft-limits (bend above a knee, untouched below) and
# also writes a solo model.wav per scenario so the model's words can
# be heard without the bed. Picks already exist (dl-daud5) — render
# only. Voices resolve per scenario as before. FULL LOG.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO

BATCH=$DL/runs/synth-dense1/audio

cd "$REPO"

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m evals.render_model_audio --picks $DL/runs/picks-dense-pt3 \
    --batch "$BATCH" --out $DL/runs/model-audio-dense-pt3
echo "dense-pt3 re-render done"
