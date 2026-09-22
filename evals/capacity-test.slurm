#!/bin/bash
#SBATCH --job-name=dl-capacty
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/dl-capacty-%j.out

# CAPACITY TEST (Codex F5-2, owner-approved 2026-09-22): can the 10-row
# approach learn 4 short conversations at all, vs a small trainable
# audio adapter (~1M params vs ~41k)? Deliberate OVERFIT probe —
# capacity, not generalisation. Both arms: same base model, same 4
# conversations, same steps/lr, unweighted CE. Pass bar (free-run,
# self-history): >=90% exact word-onset hits on the trained 4 with
# chatter <10% of ticks. Held-out 4 reported for the gap.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards
BATCH=$DL/runs/synth-big7/audio
TRAIN4="timed_task-001,timed_task-004,barge_in-000,barge_in-004"
HELD4="timed_task-008,timed_task-013,barge_in-012,barge_in-016"

cd "$REPO"

echo "===== ARM A: rows-only (41k params), 4 conversations ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps 400 --lessons "$TRAIN4" --out $DL/runs/cap-rows

echo "===== ARM B: adapter (1M params), same 4 conversations ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps 400 --lessons "$TRAIN4" --adapter --out $DL/runs/cap-adapter

for ARM in cap-rows cap-adapter; do
  for SET in "trained $TRAIN4" "heldout $HELD4"; do
    set -- $SET
    TAG=$1; IDS=$2
    echo "===== free-run $ARM on $TAG conversations ====="
    apptainer exec --nv --bind /projects,/lus,/home \
        --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
        /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
        python -m layera.freerun_diag --batch "$BATCH" \
        --ckpt $DL/runs/$ARM/p1_checkpoint.pt --shards "$SHARDS" \
        --scenarios "$IDS" --max-ticks 900 \
        2>&1 | grep -E "freerun|gold |WORD ONSETS"
  done
done
echo "capacity test done"
