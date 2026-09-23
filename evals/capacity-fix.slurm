#!/bin/bash
#SBATCH --job-name=dl-capfix
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/dl-capfix-%j.out

# REPAIR of 6792730: both arms TRAINED (checkpoints saved); arm A's two
# free-runs completed (word onsets 98.5% exact trained / 97.1% held-out,
# but chatter 33-88% of silence ticks); arm B's free-run crashed at the
# adapter application and the pipe grep ATE THE ERROR MESSAGE. This job
# re-runs ONLY the cap-adapter free-runs with FULL output — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards
BATCH=$DL/runs/synth-big7/audio
TRAIN4="timed_task-001,timed_task-004,barge_in-000,barge_in-004"
HELD4="timed_task-008,timed_task-013,barge_in-012,barge_in-016"
CKPT=$DL/runs/cap-adapter/p1_checkpoint.pt

cd "$REPO"

for SET in "trained $TRAIN4" "heldout $HELD4"; do
  set -- $SET
  TAG=$1; IDS=$2
  echo "===== free-run cap-adapter on $TAG conversations (FULL LOG) ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt "$CKPT" --shards "$SHARDS" \
      --scenarios "$IDS" --max-ticks 900
done
echo "cap-adapter free-runs done"
