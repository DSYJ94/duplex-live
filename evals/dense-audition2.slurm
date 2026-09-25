#!/bin/bash
#SBATCH --job-name=dl-daud2
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-daud2-%j.out

# AUDITION re-render for dense-sh1 with MATCHED VOICES (owner note
# 2026-09-25: the first render used the MiniCPM-o bundled demo clip
# system_ref_audio_2.wav as the cloning reference — a deep voice that
# sounded "weird" in every sample). Renderer now clones each scenario's
# model words from that lesson's own ASSISTANT voice (LibriSpeech pool:
# ls2094 / ls8224 / ls4446 / ls5142), so it sounds like the person who
# was supposed to be speaking. Picks already exist from 6861441 —
# render only, no free-run.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO

BATCH=$DL/runs/synth-dense1/audio

cd "$REPO"

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m evals.render_model_audio --picks $DL/runs/picks-dense-sh1 \
    --batch "$BATCH" --out $DL/runs/model-audio-dense-sh1
echo "matched-voice re-render done"
