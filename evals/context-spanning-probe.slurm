#!/bin/bash
#SBATCH --job-name=dl-context-probe
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-context-probe-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${CONTEXT_ASSETS_JOB:?Pinned public assets required}"
: "${CONTEXT_PRIVATE_VALUE:?Declared private development world required}"
[[ "$CONTEXT_ASSETS_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
EXTRA=()
if [ -n "${CONTEXT_EVENT_LOG_MODE:-}" ]; then EXTRA+=(--event-log-mode "$CONTEXT_EVENT_LOG_MODE"); fi
if [ -n "${CONTEXT_BACKEND_REFERENCE_PROFILE:-}" ]; then EXTRA+=(--backend-reference-profile "$CONTEXT_BACKEND_REFERENCE_PROFILE"); fi
if [ -n "${CONTEXT_RET_LOGIT_BIAS:-}" ]; then EXTRA+=(--ret-logit-bias "$CONTEXT_RET_LOGIT_BIAS"); fi
if [ -n "${CONTEXT_PERSONA_PROFILE:-}" ]; then EXTRA+=(--persona-profile "$CONTEXT_PERSONA_PROFILE"); fi
if [ -n "${CONTEXT_LOOKUP_DELIVERY_DELAY_S:-}" ]; then EXTRA+=(--lookup-delivery-delay-s "$CONTEXT_LOOKUP_DELIVERY_DELAY_S"); fi
if [ -n "${CONTEXT_EXPECTED_KIND:-}" ]; then EXTRA+=(--expected-kind "$CONTEXT_EXPECTED_KIND"); fi
if [ "${CONTEXT_MATH_ATTENTION:-0}" = 1 ]; then EXTRA+=(--math-attention); fi
if [ "${CONTEXT_BACKEND_MODE:-scheduled}" = native-asr ]; then
 EXTRA+=(--backend-mode native-asr --asr-python "$DL/envs/moshi-annotation-7379026/bin/python")
fi
if [ "${CONTEXT_BACKEND_MODE:-scheduled}" = observe-only ]; then
 EXTRA+=(--backend-mode observe-only)
fi
if [ "${CONTEXT_WARM_AUDIO:-0}" = 1 ]; then EXTRA+=(--warm-audio); fi
if [ "${CONTEXT_PENDING_STATUS:-0}" = 1 ]; then EXTRA+=(--pending-status); fi
if [ "${CONTEXT_TRAIN_HISTORY:-0}" = 1 ]; then EXTRA+=(--train-history); fi
if [ -n "${CONTEXT_ADAPTER_JOB:-}" ]; then
 [[ "$CONTEXT_ADAPTER_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--adapter-proof "$DL/runs/context-training-proof-$CONTEXT_ADAPTER_JOB")
 if [ "${CONTEXT_ADAPTER_DISABLED:-0}" = 1 ]; then EXTRA+=(--adapter-disabled); fi
 if [ "${CONTEXT_ADAPTER_ZERO:-0}" = 1 ]; then EXTRA+=(--adapter-zero); fi
fi
if [ -n "${CONTEXT_NATIVE_JOINT_JOB:-}" ]; then
 [[ "$CONTEXT_NATIVE_JOINT_JOB" =~ ^[0-9]+$ ]]
 test -z "${CONTEXT_ADAPTER_JOB:-}"
 EXTRA+=(--adapter-proof "$DL/runs/native-assistant-joint-$CONTEXT_NATIVE_JOINT_JOB")
fi
if [ -n "${CONTEXT_LAYOUT_AUDIT_JOB:-}" ]; then
 [[ "$CONTEXT_LAYOUT_AUDIT_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--layout-audit "$DL/runs/context-training-proof-$CONTEXT_LAYOUT_AUDIT_JOB/history-audit.json")
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.context_spanning_probe \
 --assets "$DL/runs/context-spanning-assets-$CONTEXT_ASSETS_JOB/manifest.json" \
 --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
 --data "$DL/runs/recovery-data-7407554" --private-value "$CONTEXT_PRIVATE_VALUE" \
 --scenario "${CONTEXT_SCENARIO:-recovery-validation-000}" \
 --voice "${CONTEXT_VOICE:-released}" \
 --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json" \
 --out "$DL/runs/context-spanning-probe-${SLURM_JOB_ID}" "${EXTRA[@]}"
