#!/bin/bash
#SBATCH --job-name=dl-sh4b
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-sh4b-%j.out

# SELF-HISTORY PROBE 2 (cap-sh2): probe 1 (cap-sh, jobs 6816545/6820600)
# proved the exposure lever moves free-run silence harder than anything
# before (sil acc 58-96%) but unweighted it lands on the MUTE pole
# (exact onsets ~1.5%). One-lever delta here: + --word-weight 3, the
# ladder's speech lever. Same rig otherwise (400 steps, lr 1e-3,
# timing-weight 50 default as in every capacity arm).
# Pass bar (free-run, trained 4): >=90% exact word-onset hits AND
# chatter <10% of silence ticks. Held-out 4 informational. FULL LOG —
# no grep pipes (they ate a traceback in 6792730).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards
BATCH=$DL/runs/synth-big7/audio
TRAIN4="timed_task-001,timed_task-004,barge_in-000,barge_in-004"
HELD4="timed_task-008,timed_task-013,barge_in-012,barge_in-016"

cd "$REPO"

echo "===== train: rows-only + self-history (50%% of steps), 4 conversations ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps 400 --lessons "$TRAIN4" --self-history 0.5 --sh-refresh 8 --word-weight 3 \
    --out $DL/runs/cap-sh2

for SET in "trained $TRAIN4" "heldout $HELD4"; do
  set -- $SET
  TAG=$1; IDS=$2
  echo "===== free-run cap-sh2 on $TAG conversations ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/cap-sh2/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$IDS" --max-ticks 900
done
echo "self-history probe done"
