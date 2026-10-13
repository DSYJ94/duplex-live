#!/bin/bash
#SBATCH --job-name=dl-native-content
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/dl-native-content-%j.out

# Plain words: six native text/audio answers with real generated speech.
# Maximum allocation: 0.75 GPU-hours; no training, no chained submissions.
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
export HF_HOME="$DL/models/hf-cache" TMPDIR="$DL/tmp"
export PYTHONPATH="$DL/pylibs-new:$DL/repo"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi
git rev-parse HEAD
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH="$PYTHONPATH" --env HF_HOME="$HF_HOME" \
    --env TMPDIR="$TMPDIR" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$DL/pytorch-new.sif" python -u -m evals.native_content_probe \
    --batch "$DL/runs/synth-scale1-q/audio" --scenario dense_dialog-001 \
    --limit 3 --out "$DL/runs/native-content-${SLURM_JOB_ID}"
