#!/bin/bash
#SBATCH --job-name=dl-lychee-env
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-lychee-env-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
OUT="$DL/envs/lychee-probe-${SLURM_JOB_ID}"
mkdir "$OUT"
# Reuse the previously verified ARM dependencies, preserving container Torch.
# This isolated overlay changes only the HF interfaces required by native source.
apptainer exec --bind /projects,/lus,/home "$DL/pytorch-new.sif" \
  python -m pip install --no-deps --only-binary=:all: --disable-pip-version-check \
  --target "$OUT" 'transformers==4.57.6' 'tokenizers==0.22.2' 'peft==0.17.1' 'accelerate==1.10.1'
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$OUT:$DL/tmp/minicpm-recovery-7377673:$DL/repo" \
  --env PYTHONDONTWRITEBYTECODE=1 "$DL/pytorch-new.sif" \
  python -c 'import torch, torchaudio, transformers, peft, datasets, librosa, s3tokenizer, onnxruntime, hyperpyyaml; print("Torch",torch.__version__,torch.__file__); print("Transformers",transformers.__version__); print("ARM dependency imports passed")'
