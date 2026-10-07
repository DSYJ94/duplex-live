#!/bin/bash
#SBATCH --job-name=dl-dechyg
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-dechyg-%j.out

# DECODE-HYGIENE TEST (2026-10-08, survey rec #1): is the free-run digit
# soup degenerate DECODING rather than a missing skill? Zero training —
# the same q2 checkpoint, four decode arms on TRAIN4, judged by the
# FREE-RUN WORD WER content metric. If sampling alone improves WER, the
# model already knows more than argmax ever showed.

set -euo pipefail
DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif
BATCH=$DL/runs/synth-scale1-q/audio
SHARDS=$DL/runs/shards-scale1-q2
CKPT=$DL/runs/scale1-q2-pt1/p1_checkpoint.pt
TRAIN4="dense_dialog-001,dense_dialog-010,dense_barge-001,dense_barge-004"

cd "$REPO"
run () {
  TAG="$1"; shift
  echo "===== decode arm: $TAG ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      $CONTAINER \
      python -m layera.freerun_diag --batch "$BATCH" --ckpt "$CKPT" \
      --shards "$SHARDS" --scenarios "$TRAIN4" "$@" \
      || echo "[hygiene] arm $TAG failed (output above is the result)"
}
run argmax-baseline
run sampled-top-p090 --top-p 0.90 --temperature 0.8
run sampled-full-hygiene --top-p 0.90 --temperature 0.8 --rep-pen 1.3 --no-junk
run full-hygiene-greedy --rep-pen 1.3 --no-junk
echo "decode hygiene done"
