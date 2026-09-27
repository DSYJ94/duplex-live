#!/bin/bash
#SBATCH --job-name=dl-dense6
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-dense6-%j.out

# PROBE 10 (dense-pt4): CURRICULUM STEP 2 — exposure after placement.
# Probe 9 (dense-pt3, pure teacher forcing) solved placement: word acc
# 0.70-1.00 trained, 0.74-0.98 held-out, onsets 5/5 5/5 4/4 3/3 exact
# — but chat fired words through every quiet tick (sil acc ~0.02).
# Reading: with SH 0 the model has never seen its own words in
# context, so no self-consistent "after I speak, hush" policy exists.
# This probe resumes the dense-pt3 rows (--init-ckpt, new train_p1
# flag) and turns SH 0.5 back on for 200 steps: exposure AFTER
# placement forms. SH corrupted probes 7/8 because speech wasn't
# learned yet — the hypothesis is the ORDER was the problem, not the
# mechanism. Everything else unchanged (v2 shards-dense2, ww 3.0,
# timing-weight 50). Pass bar (v2): >=90% exact run-onsets AND
# false speech <10% of quiet ticks, trained 4. FULL LOG — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"

cd "$REPO"

echo "===== train: curriculum step 2 (init dense-pt3, SH 0.5, 200 steps) ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps 200 --lessons "$TRAIN4" --self-history 0.5 --sh-refresh 8 \
    --word-weight 3.0 \
    --init-ckpt $DL/runs/dense-pt3/p1_checkpoint.pt \
    --out $DL/runs/dense-pt4

for SET in "trained $TRAIN4" "heldout $HELD4"; do
  set -- $SET
  TAG=$1; IDS=$2
  echo "===== free-run dense-pt4 on $TAG dense conversations ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-pt4/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$IDS" --max-ticks 900
done
echo "dense probe 10 done"
