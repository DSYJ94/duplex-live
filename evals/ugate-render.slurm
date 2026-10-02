#!/bin/bash
#SBATCH --job-name=dl-ugr
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/dl-ugr-%j.out

# Audio render of the UGATE runs (2026-10-02) so the verdict can be
# HEARD, not just read (standing rule: every voice-sample result goes
# through viz.listen). Picks already saved by job 7019984:
#   runs/scale1-ugate/picks-hold5      (HELD4, user-gate hold=5)
#   runs/scale1-ugate/picks-train4-hold5
# The panel FAILED the held config (3/4 gates: dialog-116 placement
# crash, barge-113 NEW full-mute) — the ear check confirms whether the
# mutes/suppressions are real silence and whether the surviving dialog
# speech is lexical. Renders only; no model inference.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

BATCH=$DL/runs/synth-scale1/audio

cd "$REPO"

for ARM in held5:scale1-ugate/picks-hold5:model-audio-ugate-held \
           train5:scale1-ugate/picks-train4-hold5:model-audio-ugate-train; do
  IFS=: read -r TAG PICKS OUT <<< "$ARM"
  echo "===== render ugate $TAG ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO \
      --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m evals.render_model_audio --picks "$DL/runs/$PICKS" \
      --batch "$BATCH" --out "$DL/runs/$OUT"
done
echo "ugate renders done"
