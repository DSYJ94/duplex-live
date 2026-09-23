#!/bin/bash
#SBATCH --job-name=dl-sh4
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-sh4-%j.out

# SELF-HISTORY PROBE (exposure fix, owner-approved 2026-09-23): the
# capacity test (6792730/6814321) showed neither rows-only nor adapter
# learns 4 conversations free-run — both trained TEACHER-FORCED and were
# graded FREE-RUN. This probe interleaves self-history steps (every 2nd
# step): the model free-runs the lesson against itself (no grad), then
# trains with its OWN emission tokens as input and GOLD labels — it
# learns to recover from its own chatter. Rows-only recipe otherwise
# identical to capacity arm A (400 steps, unweighted CE, lr 1e-3).
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
    --steps 400 --lessons "$TRAIN4" --self-history 0.5 --sh-refresh 8 \
    --out $DL/runs/cap-sh

for SET in "trained $TRAIN4" "heldout $HELD4"; do
  set -- $SET
  TAG=$1; IDS=$2
  echo "===== free-run cap-sh on $TAG conversations ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/cap-sh/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$IDS" --max-ticks 900
done
echo "self-history probe done"
