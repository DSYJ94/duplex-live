#!/bin/bash
#SBATCH --job-name=dl-dgaud
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/dl-dgaud-%j.out

# GATED AUDITION (pre-agreed branch after gate1 half-worked, 2026-09-28):
# render pt3+gate for the gallery so the result is audible. Two arms on
# the FULL trained lessons (no tick cap):
#   A: gate only, threshold 0.35 — the literal pt3+gate render
#      (dialog-001 is expected fully muted; that failure is audible too)
#   B: combo decode — gate 0.35 + the approved stopgap rules
#      (--min-gap-ticks 4 --user-gate --gate-hold 3), the candidate
#      live-loop config
# freerun_diag now also prints the gate's own P at the gold speech
# onsets (gate1 diagnosis: dialog gold onsets score below chatter
# states on other lessons -> full mute; the print verifies it
# tick-by-tick). PYTHONPATH IS PER-STAGE: the TTS overlay
# pylibs-minicpm vendors its own transformers (no
# Qwen3OmniMoeThinkerForConditionalGeneration) — exported job-wide it
# shadows the real one and killed 6913565 at 18 s; freerun gets
# pylibs:$REPO, only the render stage gets the overlay. FULL LOG —
# no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
FREE_PP=$DL/pylibs:$REPO
TTS_PP=$DL/pylibs-minicpm:$DL/pylibs:$REPO

BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
CKPT=$DL/runs/dense-pt3/p1_checkpoint.pt
GATE=$DL/runs/dense-gate1/gate_checkpoint.pt

cd "$REPO"

run_arm () {
  local NAME=$1; shift
  echo "===== free-run pt3+gate $NAME (full TRAIN4, save picks) ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$FREE_PP --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt "$CKPT" --shards $DL/runs/shards-dense2 \
      --scenarios "$TRAIN4" --gate-ckpt "$GATE" --gate-threshold 0.35 \
      --save-picks $DL/runs/picks-$NAME "$@"
  echo "===== render $NAME to audio ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$TTS_PP --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m evals.render_model_audio --picks $DL/runs/picks-$NAME \
      --batch "$BATCH" --out $DL/runs/model-audio-$NAME
}

run_arm dense-pt3-gate035
run_arm dense-pt3-gate-combo --min-gap-ticks 4 --user-gate --gate-hold 3
echo "dense gate audition done"
