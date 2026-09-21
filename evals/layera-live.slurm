#!/bin/bash
#SBATCH --job-name=layera-live
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/layera-live-%j.out
#
# Eval harness, Layer-A edition (Issue #7 gate): drive the Phase-1
# duplex thinker through the T-protocol under the main pylibs overlay,
# then score the recordings with the same metrics as the baselines.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

CKPT=${CKPT:-$DL/runs/p1-v3/p1_checkpoint.pt}
OUT=${OUT:-$DL/runs/harness-layera-$(basename $(dirname "$CKPT"))}
TESTS=${TESTS:-t1,t2,t3,t4,t5}

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m evals.harness.run --live layera \
    --ckpt "$CKPT" --tests "$TESTS" \
    --out "$OUT" --prompts $DL/runs/bakeoff/prompts

echo "[layera-live] recordings + report: $OUT"
