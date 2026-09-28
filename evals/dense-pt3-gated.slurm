#!/bin/bash
#SBATCH --job-name=dl-dgate
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/dl-dgate-%j.out

# STOPGAP AUDITION (owner-approved 2026-09-27, "decode-rule stopgap,
# then listen"): dense-pt3 (probe 9 — perfect placement, constant
# chatter) replayed with two hard decode rules gating word-FIRST picks
# only; a word already in flight always completes:
#   --min-gap-ticks 4  (gold word onsets run ~4.5 ticks apart; decode
#                       chatter fires word-first nearly every tick)
#   --user-gate        (block onsets while in-slot energy says the
#                       user is speaking + --gate-hold 3 ticks tail;
#                       tightest gold user->assistant gap is 400 ms,
#                       so hold 3 keeps every legitimate first onset)
# Then TTS the gated picks -> runs/model-audio-dense-pt3-gated for the
# gallery. This is a demo patch, not Layer A — hand-coded turn-taking
# is what the model should learn; the training-side fix is probe 11.
# FULL LOG — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"

cd "$REPO"

echo "===== free-run dense-pt3 with stopgap decode rules ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$TRAIN4" --max-ticks 900 \
    --min-gap-ticks 4 --user-gate --gate-hold 3 \
    --save-picks $DL/runs/picks-dense-pt3-gated

echo "===== TTS the gated picks ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m evals.render_model_audio --picks $DL/runs/picks-dense-pt3-gated \
    --batch "$BATCH" --out $DL/runs/model-audio-dense-pt3-gated
echo "gated audition done"
