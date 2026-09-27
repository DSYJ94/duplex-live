#!/bin/bash
#SBATCH --job-name=dl-dense4
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-dense4-%j.out

# PROBE 8 (dense-pt2): v2 labels + word-weight 3.0 — the MUTING
# ENDPOINT probe (owner strategy 2026-09-27: bounded ladder + M2 in
# parallel). Same rig as probe 7 (dense-pt1: v2 shards-dense2, SH 0.5,
# refresh 8, timing-weight 50, trained 4 / held 4 dense lessons).
# ONE variable vs probe 7: ww 1.0 -> 3.0 — the only dose that ever
# produced a speaking model (probes 2/5 on v1), now pushing on v2's
# 60%-speech labels. Outcome decides the branch:
#   speaks  -> probe ladder moves to PLACEMENT (problem 2);
#   mutes   -> ww lever exhausted on both contracts -> next is SH 0
#              (teacher forcing), and if that fails, redesign instead
#              of more probes.
# Pass bar (v2): >=90% exact gold speech-run-onset hits AND false
# speech <10% of quiet ticks, trained 4. FULL LOG — no grep pipes.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"

cd "$REPO"

echo "===== train: rows-only + SH 0.5 + ww3 on V2 shards ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps 400 --lessons "$TRAIN4" --self-history 0.5 --sh-refresh 8 \
    --word-weight 3.0 \
    --out $DL/runs/dense-pt2

for SET in "trained $TRAIN4" "heldout $HELD4"; do
  set -- $SET
  TAG=$1; IDS=$2
  echo "===== free-run dense-pt2 on $TAG dense conversations ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-pt2/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$IDS" --max-ticks 900
done
echo "dense probe 8 done"
