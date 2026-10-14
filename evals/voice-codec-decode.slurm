#!/bin/bash
#SBATCH --job-name=dl-codec-decode
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-codec-decode-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
DEVICE="${CODEC_DEVICE:-cpu}"
case "$DEVICE" in cpu) GPU=() ;; cuda) GPU=(--nv) ;; *) exit 2 ;; esac
apptainer exec "${GPU[@]}" --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/personaplex/moshi:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo" \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
  "$DL/pytorch-new.sif" python -u -m evals.voice_codec_decode_probe \
  --codec "$DL/runs/voice-quality-7414595/codec" --device "$DEVICE" \
  --out "$DL/runs/voice-codec-decode-${SLURM_JOB_ID}"
