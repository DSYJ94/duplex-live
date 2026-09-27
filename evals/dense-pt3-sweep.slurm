#!/bin/bash
#SBATCH --job-name=dl-pt3swp
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-pt3swp-%j.out

# Decode-side companion to probe 10: signed --sil-bias sweep on
# dense-pt3 (probe 9's teacher-forced checkpoint: perfect placement,
# constant chatter). freerun_diag computes
#   logits[<sil>] -= sil_bias
# so NEGATIVE values ADD to the sil logit — suppress speech. Four
# suppression strengths on the 4 trained lessons map the
# placement/chatter tradeoff available at decode time with zero
# training: how much chatter dies before the exact onsets do. The
# same knob, sign-flipped, was the muting-era decode lever; the
# metric lines make the curve directly comparable. FULL LOG — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"

cd "$REPO"

for B in 0.5 1 2 4; do
  echo "===== sil-bias -$B (suppress speech) ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$TRAIN4" --max-ticks 900 --sil-bias -"$B"
done
echo "dense-pt3 sweep done"
