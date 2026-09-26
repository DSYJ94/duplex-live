#!/bin/bash
#SBATCH --job-name=dl-dense3
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-dense3-%j.out

# PROBE 7 (dense-pt1): first probe under the V2 label contract
# (per-tick speech, ADR-0004, owner-approved 2026-09-26). Same rig,
# same 4 trained + 4 held dense lessons as probes 5-6; shards rebuilt
# under v2 (shards-dense2, job dl-shard2). Recipe = probe-2 base with
# word-weight BACK TO 1.0: ww existed to compensate v1's 88%-silence
# labels; v2 speech labels are 59.7% of ticks so the compensation is
# removed (keeping ww3 would triple a balanced class — probe 6 showed
# where that leads). ONE conceptual variable vs probe 5: the labels.
# Pass bar (v2 definitions): >=90% exact gold speech-RUN-onset hits
# AND false speech on <10% of quiet ticks, trained 4. FULL LOG — no
# grep pipes.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"

cd "$REPO"

echo "===== train: rows-only + SH 0.5 + ww1 on V2 shards ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps 400 --lessons "$TRAIN4" --self-history 0.5 --sh-refresh 8 \
    --out $DL/runs/dense-pt1

for SET in "trained $TRAIN4" "heldout $HELD4"; do
  set -- $SET
  TAG=$1; IDS=$2
  echo "===== free-run dense-pt1 on $TAG dense conversations ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-pt1/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$IDS" --max-ticks 900
done
echo "dense probe 7 done"
