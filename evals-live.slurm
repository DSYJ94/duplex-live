#!/bin/bash
#SBATCH --job-name=evals-live
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/evals-live-%j.out
#
# Eval harness live run (Issue #6): drive MiniCPM-o duplex through the
# T-protocol, write replay-layout recordings, score them with the same
# metrics as the baselines. Under the pylibs-minicpm overlay (4.51).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache
export TMPDIR=$DL/tmp
export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO

OUT=${OUT:-$DL/runs/harness-live-$(date +%m%d-%H%M)}
ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m evals.harness.run --live minicpm \
    --out "$OUT" --prompts $DL/runs/bakeoff/prompts --assets "$ASSETS"

echo "[evals-live] recordings + report: $OUT"
