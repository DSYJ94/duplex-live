#!/bin/bash
#SBATCH --job-name=dl-mcpm-recovery
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-mcpm-recovery-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
OVERLAY="$DL/tmp/minicpm-recovery-${SLURM_JOB_ID}"
mkdir "$OVERLAY"
# Preserve all existing environments. Reuse pure-Python dependencies while
# preventing the old overlay's Torch/CUDA packages and metadata from winning.
for source in "$DL/pylibs-minicpm/"*; do
    name=$(basename "$source")
    case "$name" in
        torch|torch-*|torchgen|functorch|nvidia|nvidia_*|triton|triton-*|torchvision*|torchaudio*|__pycache__) continue ;;
    esac
    ln -s "$source" "$OVERLAY/$name"
done
# Token2wav needs torchaudio. Install only its matching-version ARM wheel into
# this disposable overlay; --no-deps cannot replace container Torch/CUDA.
apptainer exec --bind /projects,/lus,/home "$DL/pytorch-new.sif" \
    python -m pip install --no-deps --only-binary=:all: --disable-pip-version-check \
    --target "$OVERLAY" 'torchaudio==2.9.0'
# Matching TorchCodec 0.9 has no ARM wheel in the CPU index. The probe uses
# explicit SoundFile WAV loading; TorchAudio resampling and features are intact.
MODEL="$DL/models/hf-cache/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"
git rev-parse HEAD
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH="$OVERLAY:$DL/repo" --env HF_HOME="$DL/models/hf-cache" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$DL/pytorch-new.sif" python -u -m evals.minicpm_recovery_probe \
    --batch "$DL/runs/synth-scale1-q/audio" --scenario dense_dialog-001 \
    --model "$MODEL" --seconds 60 --out "$DL/runs/minicpm-recovery-${SLURM_JOB_ID}"
