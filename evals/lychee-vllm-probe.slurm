#!/bin/bash
#SBATCH --job-name=dl-lychee-runtime
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-lychee-runtime-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
TASK_SOURCE="${DL_TASK_SOURCE:-$DL/repo}"
if [ "$TASK_SOURCE" != "$DL/repo" ]; then
 : "${DL_TASK_COMMIT:?Immutable worktree commit required}"
 test "$(git -C "$TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
 test -z "$(git -C "$TASK_SOURCE" status --porcelain)"
fi
cd "$TASK_SOURCE"
: "${LYCHEE_PREFLIGHT_JOB:?Actual CUDA kernel and Python import proof required}"
[[ "$LYCHEE_PREFLIGHT_JOB" =~ ^[0-9]+$ ]]
DEPS=""
if [ -n "${LYCHEE_RUNTIME_DEPS_JOB:-}" ]; then
 [[ "$LYCHEE_RUNTIME_DEPS_JOB" =~ ^[0-9]+$ ]]
 DEPS="$DL/envs/lychee-vllm-deps-$LYCHEE_RUNTIME_DEPS_JOB"
 test -f "$DL/runs/lychee-vllm-deps-$LYCHEE_RUNTIME_DEPS_JOB/report.json"
fi
RUNTIME="$DL/envs/lychee-vllm-$LYCHEE_PREFLIGHT_JOB"
EXTRA=()
if [ "${LYCHEE_WARM_INPUT:-0}" = 1 ]; then EXTRA+=(--warm-input); fi
if [ "${LYCHEE_FORWARD_CAPTURE:-0}" = 1 ]; then EXTRA+=(--forward-capture); fi
if [ -n "${LYCHEE_SEMANTIC_JOB:-}" ]; then
 [[ "$LYCHEE_SEMANTIC_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--semantic-adapter "$DL/runs/lychee-semantic-proof-$LYCHEE_SEMANTIC_JOB/adapter.pt"
   --semantic-proof "$DL/runs/lychee-semantic-proof-$LYCHEE_SEMANTIC_JOB/report.json")
fi
if [ "${LYCHEE_WORKER_DIAG:-0}" = 1 ]; then
 : "${LYCHEE_SCENARIO:?Private development scenario required}"
 set -a
 source "$DL/keys/openai.env"
 set +a
 EXTRA+=(--worker-context --data "${LYCHEE_WORKER_DATA:-$DL/runs/recovery-data-7407554}" --scenario "$LYCHEE_SCENARIO")
 if [ "${LYCHEE_COLLECT_TRAIN_STATE:-0}" = 1 ]; then EXTRA+=(--collect-training-state); fi
 if [ -n "${LYCHEE_RESULT_RESPONSE:-}" ]; then EXTRA+=(--result-response "$LYCHEE_RESULT_RESPONSE"); fi
 if [ -n "${LYCHEE_RESULT_DELIVERY:-}" ]; then EXTRA+=(--result-delivery "$LYCHEE_RESULT_DELIVERY"); fi
 if [ "${LYCHEE_STATIC_CONTEXT:-0}" = 1 ]; then EXTRA+=(--static-context-diagnostic); fi
 if [ -n "${LYCHEE_PRIVATE_VALUE:-}" ]; then EXTRA+=(--private-value "$LYCHEE_PRIVATE_VALUE"); fi
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$RUNTIME:$DEPS:$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.lychee_vllm_probe \
 --assets "$DL/runs/lychee-assets-7414609/manifest.json" \
 --batch "$DL/runs/synth-scale1-q/audio" --runtime "$RUNTIME" \
 --preflight "$DL/runs/lychee-vllm-preflight-$LYCHEE_PREFLIGHT_JOB" \
 --out "$DL/runs/lychee-vllm-probe-${SLURM_JOB_ID}" "${EXTRA[@]}"
