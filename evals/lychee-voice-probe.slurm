#!/bin/bash
#SBATCH --job-name=dl-lychee-voice
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-lychee-voice-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
: "${LYCHEE_ASSETS_JOB:?Pinned model download job required}"
: "${LYCHEE_ENV_JOB:?Isolated environment job required}"
: "${VOICE_REFERENCE:?Synthetic development voice required}"
: "${VOICE_PROVENANCE:?Voice receipt required}"
[[ "$LYCHEE_ASSETS_JOB" =~ ^[0-9]+$ && "$LYCHEE_ENV_JOB" =~ ^[0-9]+$ ]]
EXTRA=()
if [ "${LYCHEE_PACED:-0}" = 1 ]; then EXTRA=(--paced); fi
if [ "${LYCHEE_ENCODER_MEMO:-0}" = 1 ]; then EXTRA+=(--encoder-memo); fi
if [ "${LYCHEE_PROFILE:-0}" = 1 ]; then EXTRA+=(--profile); fi
if [ "${LYCHEE_NO_CUDNN_SDPA:-0}" = 1 ]; then EXTRA+=(--no-cudnn-sdpa); fi
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/envs/lychee-probe-$LYCHEE_ENV_JOB:$DL/tmp/minicpm-recovery-7377673:$DL/repo" \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" python -u -m evals.lychee_voice_probe \
  --assets "$DL/runs/lychee-assets-$LYCHEE_ASSETS_JOB/manifest.json" \
  --batch "$DL/runs/synth-scale1-q/audio" \
  --voice "$VOICE_REFERENCE" --voice-provenance "$VOICE_PROVENANCE" \
  --out "$DL/runs/lychee-voice-${SLURM_JOB_ID}" "${EXTRA[@]}"
