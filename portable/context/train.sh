#!/usr/bin/env bash
# Run on a destination NVIDIA GPU host. This does not provision instances.
set -euo pipefail
: "${DL_RESTORED_TREE:?Host directory containing the restored Isambard tree}"
: "${DL_TASK_COMMIT:?Exact immutable training commit}"
: "${DL_RUN_ID:?New numeric run identifier, distinct from existing runs}"
: "${CONTEXT_ACTUAL_HEAD_JOBS:?Space-separated approved early capture identifiers}"
[[ "$DL_TASK_COMMIT" =~ ^[a-f0-9]{40}$ && "$DL_RUN_ID" =~ ^[0-9]+$ ]]
ROOT=/projects/u6si/duplex-live
SOURCE="$ROOT/worktrees/recovery-$DL_TASK_COMMIT"
HOST_SOURCE="$DL_RESTORED_TREE/worktrees/recovery-$DL_TASK_COMMIT"
test "$(git -C "$HOST_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$HOST_SOURCE" status --porcelain)"
test ! -e "$DL_RESTORED_TREE/runs/context-training-proof-$DL_RUN_ID"
ARGS=(--assets "$ROOT/runs/context-spanning-assets-7454205/manifest.json"
 --base "$ROOT/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
 --voice-reference "$ROOT/runs/voice-quality-7414595/codec/reference.wav"
 --voice-provenance "$ROOT/runs/voice-quality-7414595/codec/report.json"
 --cache "$ROOT/runs/recovery-mixed-codes-7411513"
 --cache "$ROOT/runs/recovery-codes-7434926" --cache "$ROOT/runs/recovery-codes-7435538"
 --actual-source-data "$ROOT/runs/recovery-data-7407554"
 --request-policy native-prelead-epad --out "$ROOT/runs/context-training-proof-$DL_RUN_ID")
for J in $CONTEXT_ACTUAL_HEAD_JOBS; do
 [[ "$J" =~ ^[0-9]+$ ]]
 ARGS+=(--actual-early-data "$ROOT/runs/context-training-proof-$J")
done
MODE="${CONTEXT_MODE:-audit}"
case "$MODE" in
 audit) ARGS+=(--actual-temporal-audit) ;;
 fit|resume)
  : "${CONTEXT_ACTUAL_TEMPORAL_PREFLIGHT_JOB:?Passed independent temporal audit required}"
  [[ "$CONTEXT_ACTUAL_TEMPORAL_PREFLIGHT_JOB" =~ ^[0-9]+$ ]]
  ARGS+=(--actual-temporal-fit --actual-temporal-preflight "$ROOT/runs/context-training-proof-$CONTEXT_ACTUAL_TEMPORAL_PREFLIGHT_JOB")
  if [ "$MODE" = resume ]; then
   : "${CONTEXT_RESUME_TRAINING_JOB:?Verified parent checkpoint identifier}"
   [[ "$CONTEXT_RESUME_TRAINING_JOB" =~ ^[0-9]+$ ]]
   ARGS+=(--resume-training "$ROOT/runs/context-training-proof-$CONTEXT_RESUME_TRAINING_JOB/training-state.pt")
  fi
  ARGS+=(--actual-fit-updates "${CONTEXT_ACTUAL_FIT_UPDATES:-8}") ;;
 *) echo 'CONTEXT_MODE must be audit, fit or resume' >&2; exit 2 ;;
esac
run() {
 docker run --rm --gpus '"device=0"' --ipc=host \
  --user "$(id -u):$(id -g)" \
  -v "$DL_RESTORED_TREE:$ROOT" -w "$SOURCE" -e PYTHONPATH="$SOURCE" \
  -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/duplex-triton \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/duplex-inductor -e HF_HOME="$ROOT/models/hf-cache" \
  "${DL_IMAGE:-duplex-context:25.10}" "$@"
}
run -m unittest evals.harness.test_context_training_state
run -m evals.context_training_proof "${ARGS[@]}"
if [ "$MODE" != audit ]; then
 run -m evals.context_training_proof "${ARGS[@]}" --reload
fi
