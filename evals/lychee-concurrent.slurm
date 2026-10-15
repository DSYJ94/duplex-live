#!/bin/bash
#SBATCH --job-name=dl-lychee-concurrent
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-lychee-concurrent-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Clean immutable worktree required}"
: "${DL_TASK_COMMIT:?Exact immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
EXTRA=()
if [ -n "${LYCHEE_SEMANTIC_JOB:-}" ]; then
 [[ "$LYCHEE_SEMANTIC_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--semantic-adapter "$DL/runs/lychee-semantic-proof-$LYCHEE_SEMANTIC_JOB/adapter.pt"
  --semantic-proof "$DL/runs/lychee-semantic-proof-$LYCHEE_SEMANTIC_JOB/report.json")
 EXTRA+=(--semantic-activation "${LYCHEE_SEMANTIC_ACTIVATION:-always}")
fi
if [ "${LYCHEE_WORKER_DIAG:-0}" = 1 ]; then
 : "${LYCHEE_SCENARIO:?Actual-worker scenario required}"
 set -a
 source "$DL/keys/openai.env"
 set +a
 EXTRA+=(--worker-context --data "$DL/runs/recovery-data-7407554" --scenario "$LYCHEE_SCENARIO"
  --result-response "${LYCHEE_RESULT_RESPONSE:-model}")
 if [ -n "${LYCHEE_PRIVATE_VALUE:-}" ]; then EXTRA+=(--private-value "$LYCHEE_PRIVATE_VALUE"); fi
fi
RUNTIME="$DL/envs/lychee-vllm-7441739"
DEPS="$DL/envs/lychee-vllm-deps-7441737"
LM="$RUNTIME:$DEPS:$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$DL_TASK_SOURCE"
TTS="$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$DL_TASK_SOURCE"
apptainer exec --nv --bind /projects,/lus,/home \
 --env DL_LM_PYTHONPATH="$LM" --env DL_VOCODER_PYTHONPATH="$TTS" \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env HF_HOME="$DL/models/hf-cache" \
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.lychee_concurrent \
 --assets "$DL/runs/lychee-assets-7414609/manifest.json" \
 --batch "$DL/runs/synth-scale1-q/audio" --runtime "$RUNTIME" \
 --preflight "$DL/runs/lychee-vllm-preflight-7441739" \
 --voice "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json" \
 --out "$DL/runs/lychee-concurrent-${SLURM_JOB_ID}" "${EXTRA[@]}"
