#!/bin/bash
#SBATCH --job-name=dl-voice-quality
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-voice-quality-%j.out
# Finite, inference-only development experiment; no optimizer or data approval.
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
SOURCE="$DL/personaplex"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 3428dfd95309a7f3c84fd93259ded0f810d1ff91
MODEL="$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
OUT="$DL/runs/voice-quality-${SLURM_JOB_ID}"
mkdir "$OUT"
run_python() {
  apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH="$SOURCE/moshi:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo" \
    --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
    --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
    "$DL/pytorch-new.sif" python -u "$@"
}
run_python -m evals.voice_codec_probe --source "$SOURCE" --model "$MODEL" \
  --reference-run "$DL/runs/gpt-live-replay-7400991/t1" --out "$OUT/codec"
for VOICE in NATF1 NATM1 synthetic-marin; do
  EXTRA=()
  if [ "$VOICE" = synthetic-marin ]; then
    PROMPT="$OUT/codec/reference.wav"
    EXTRA=(--voice-provenance "$OUT/codec/report.json")
  else
    PROMPT="$DL/models/personaplex-voices/voices/$VOICE.pt"
  fi
  run_python -m evals.personaplex_recovery_probe \
    --batch "$DL/runs/synth-scale1-q/audio" --source "$SOURCE" --model "$MODEL" \
    --voice "$PROMPT" --out "$OUT/$VOICE" "${EXTRA[@]}"
done
