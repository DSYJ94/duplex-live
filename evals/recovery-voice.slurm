#!/bin/bash
#SBATCH --job-name=dl-recovery-voice
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-recovery-voice-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
SOURCE="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
cd "$DL/repo"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 8c6dfc101b7871baa428424bcdc583b74fb561d9
export DL SOURCE
export RECOVERY_VOICE_DATA="${RECOVERY_VOICE_DATA:?Verified conversion required}"
export RECOVERY_VOICE_SCENARIO="${RECOVERY_VOICE_SCENARIO:-recovery-validation-000}"
export RECOVERY_VOICE_ADAPTER="${RECOVERY_VOICE_ADAPTER:-}"
export RECOVERY_VOICE_IN_PROCESS="${RECOVERY_VOICE_IN_PROCESS:-1}"
if [ -n "$RECOVERY_VOICE_ADAPTER" ] && [ ! -f "$RECOVERY_VOICE_ADAPTER" ]; then
  echo "RECOVERY_VOICE_ADAPTER must name the saved lora.safetensors file" >&2
  exit 2
fi
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$SOURCE/moshi:$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
PY="$DL/envs/moshi-rag-7379661/bin/python"
export TRITON_LIBCUDA_PATH
TRITON_LIBCUDA_PATH=$("$PY" -m evals.cuda_linker_path --out "$DL/runs/recovery-voice-driver-${SLURM_JOB_ID}")
ARGS=(--manifest "$DL/runs/moshi-models-7378305/model-manifest.json" \
  --tokenizer "$DL/models/hf-cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
  --out "$DL/runs/recovery-voice-encoder-${SLURM_JOB_ID}" \
  --voice-data "$RECOVERY_VOICE_DATA" --voice-out "$DL/runs/recovery-voice-${SLURM_JOB_ID}/t1" \
  --voice-scenario "$RECOVERY_VOICE_SCENARIO")
if [ -n "$RECOVERY_VOICE_ADAPTER" ]; then ARGS+=(--voice-adapter "$RECOVERY_VOICE_ADAPTER"); fi
if [ "$RECOVERY_VOICE_IN_PROCESS" = 1 ]; then ARGS+=(--in-process-voice); fi
"$PY" -u -m evals.moshi_reference_probe "${ARGS[@]}"
CONTAINER
