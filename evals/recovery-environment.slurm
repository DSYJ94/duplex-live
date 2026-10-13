#!/bin/bash
#SBATCH --job-name=dl-recovery-env
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-recovery-env-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
OUT="$DL/runs/recovery-env-${SLURM_JOB_ID}"
mkdir -p "$OUT"
git rev-parse HEAD
# No old overlays: verify the container's own binary packages first.
apptainer exec --nv --bind /projects,/lus,/home --env PYTHONPATH="$DL/repo" \
    "$DL/pytorch-new.sif" python -u -m evals.recovery_environment --out "$OUT/base.json"
if [ -n "${PROBE_DIR:-}" ]; then
    apptainer exec --nv --bind /projects,/lus,/home \
        --env PYTHONPATH="$DL/pylibs-new:$DL/repo" \
        --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
        "$DL/pytorch-new.sif" python -u -m evals.native_speech_audit --probe "$PROBE_DIR"
fi
