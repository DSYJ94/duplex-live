#!/bin/bash
#SBATCH --job-name=dl-dense7
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-dense7-%j.out

# PROBE 11 (dense-pt5): EXPOSURE-DISTRIBUTION REDESIGN. Probe 10 killed
# the curriculum: recollected self-histories flip-flopped between the
# two degenerate attractors (dialog-001 recollect went 1285/1344
# speech-first -> 2/1344 -> 390/450), training saw contradictory
# self-stories, and probe 9's placement eroded (exact onsets 5/5 5/5
# 4/4 3/3 -> 0/5 3/5 1/4 1/3). Redesign: --sh-mode corrupt builds the
# exposure from the GOLD schedule the way decode actually drifts —
# word-first chatter bursts in quiet stretches, a word's first piece
# bleeding early — labels stay gold, so every corrupted sample
# teaches its own recovery ("I just babbled -> hush"). Bounded (~1% of
# slots) so placement keeps its clean-gold backbone; no recollection
# loop, so runs are much faster than probe 10's 3h20. Same init
# (dense-pt3 rows), SH 0.5, 200 steps, v2 shards-dense2, ww 3.0,
# timing-weight 50. Pass bar (v2): >=90% exact run-onsets AND false
# speech <10% of quiet ticks, trained 4. FULL LOG — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"

cd "$REPO"

echo "===== train: exposure redesign (init dense-pt3, corrupt-gold SH 0.5, 200 steps) ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps 200 --lessons "$TRAIN4" --self-history 0.5 --sh-mode corrupt \
    --word-weight 3.0 \
    --init-ckpt $DL/runs/dense-pt3/p1_checkpoint.pt \
    --out $DL/runs/dense-pt5

for SET in "trained $TRAIN4" "heldout $HELD4"; do
  set -- $SET
  TAG=$1; IDS=$2
  echo "===== free-run dense-pt5 on $TAG dense conversations ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-pt5/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$IDS" --max-ticks 900
done
echo "dense probe 11 done"
