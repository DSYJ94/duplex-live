#!/bin/bash
#SBATCH --job-name=dl-shfix
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-shfix-%j.out

# REPAIR of 6816545: training COMPLETED (400/400 steps, ckpt
# runs/cap-sh) but both free-run evals died at the WORD ONSETS print —
# a metric refactor left a NameError (exact). Metrics now extracted to
# word_onset_stats() with a CPU regression test
# (evals/harness/test_freerun_metrics.py). This job re-runs ONLY the
# evals. FULL LOG — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards
BATCH=$DL/runs/synth-big7/audio
TRAIN4="timed_task-001,timed_task-004,barge_in-000,barge_in-004"
HELD4="timed_task-008,timed_task-013,barge_in-012,barge_in-016"

cd "$REPO"

# guard: the regression test must pass before any GPU time is spent
apptainer exec --bind /projects \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m evals.harness.test_freerun_metrics

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
echo "cap-sh evals done"
