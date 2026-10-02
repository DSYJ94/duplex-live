#!/bin/bash
#SBATCH --job-name=dl-diag1
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/dl-diag1-%j.out

# D1 (2026-10-02, owner-approved): the decisive state-diagnosis on the
# SAVED probe-1 rollout buffer (4,374 states). No 30B model is loaded —
# probes + tokenizer only, minutes of compute. Arbitrates between
# "signal present nonlinearly (LoRA-readout viable)" vs "signal is
# lesson-local history basins (only objective-level fixes help)".
# FULL LOG — no grep at run time.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$REPO

cd "$REPO"
echo "===== D1: state diagnosis on saved rollout buffer ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.diag_states --states $DL/runs/dagger-states/scale1.pt
echo "diag1 done"
