#!/bin/bash
#SBATCH --job-name=dl-native-v2-native-approval
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dl-native-v2-native-%A_%a.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable committed source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
: "${NATIVE_V2_CODES_JOB:?Complete native-v2 codec cache required}"
: "${NATIVE_V2_PHASE:?Choose scope, capture, approve, or math}"
[[ "$NATIVE_V2_CODES_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
CACHE="$DL/runs/native-v2-codes-$NATIVE_V2_CODES_JOB"
ASSETS="$DL/runs/context-spanning-assets-7454205/manifest.json"
BASE="$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
VOICE="$DL/runs/context-training-proof-7510331/voice.pt"
run() {
 apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env OPENAI_API_KEY= \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" python -u "$@"
}
# This is a NEW complete 24-teacher scope. Old 25-teacher approvals/guards stay intact.
run -c 'import sys; from pathlib import Path; from evals.recovery_data import validate_cache; from data.synthetic.native_curriculum_v2 import specifications; r=validate_cache(Path(sys.argv[1])); expected={"teacher-"+s["scenario_id"] for s in specifications() if s["split"]=="train"}; assert len(r["examples"])==24 and {e["scenario_id"] for e in r["examples"]}==expected and all(e["split"]=="train" for e in r["examples"])' "$CACHE"
if [ "$NATIVE_V2_PHASE" = scope ]; then
 run -m evals.context_retention_teacher --assets "$ASSETS" --tokenizer "$BASE/tokenizer_spm_32k_3.model" \
  --voice "$VOICE" --cache "$CACHE" --all-approved-teachers --observed-native-boundary \
  --out "$DL/runs/native-v2-native-scope-$SLURM_JOB_ID"
 exit 0
fi
: "${NATIVE_V2_SCOPE_JOB:?Independent complete native-v2 scope required}"
[[ "$NATIVE_V2_SCOPE_JOB" =~ ^[0-9]+$ ]]
SCOPE="$DL/runs/native-v2-native-scope-$NATIVE_V2_SCOPE_JOB/report.json"
ARGS=(--assets "$ASSETS" --base "$BASE"
 --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav"
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json"
 --cache "$CACHE" --request-policy native-prelead-epad
 --teacher-engine-input-audit "$SCOPE" --teacher-engine-input-voice "$VOICE")
if [ "$NATIVE_V2_PHASE" = capture ]; then
 : "${SLURM_ARRAY_JOB_ID:?Submit the complete 0-23 array with at most four concurrent tasks}"
 [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ && "$SLURM_ARRAY_TASK_ID" -lt 24 ]]
 ID=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["passed"] and len(r["teachers"])==24; print(r["teachers"][int(sys.argv[2])]["id"])' "$SCOPE" "$SLURM_ARRAY_TASK_ID")
 run -m evals.context_training_proof "${ARGS[@]}" --teacher-engine-input-id "$ID" \
  --out "$DL/runs/native-v2-native-capture-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID"
elif [ "$NATIVE_V2_PHASE" = approve ]; then
 : "${NATIVE_V2_CAPTURE_ARRAY_JOB:?Complete native teacher trace array required}"
 [[ "$NATIVE_V2_CAPTURE_ARRAY_JOB" =~ ^[0-9]+$ ]]
 CAPTURES=()
 for I in $(seq 0 23); do
  CAPTURES+=(--capture "$DL/runs/native-v2-native-capture-${NATIVE_V2_CAPTURE_ARRAY_JOB}_$I")
 done
 run -m evals.context_teacher_native_approval --assets "$ASSETS" \
  --tokenizer "$BASE/tokenizer_spm_32k_3.model" --voice "$VOICE" --cache "$CACHE" \
  --scope "$SCOPE" "${CAPTURES[@]}" --out "$DL/runs/native-v2-native-approval-$SLURM_JOB_ID"
elif [ "$NATIVE_V2_PHASE" = math ]; then
 : "${NATIVE_V2_APPROVAL_JOB:?Independent complete input approval required}"
 [[ "$NATIVE_V2_APPROVAL_JOB" =~ ^[0-9]+$ ]]
 EXTRA=()
 if [ -n "${NATIVE_V2_FP32_PREFLIGHT_JOB:-}" ]; then
  : "${NATIVE_V2_FP32_SOURCE_COMMIT:?Original FP32 audit source required}"
  [[ "$NATIVE_V2_FP32_PREFLIGHT_JOB" =~ ^[0-9]+$ ]]
  EXTRA+=(--teacher-native-fp32-preflight "$DL/runs/native-v2-native-math-$NATIVE_V2_FP32_PREFLIGHT_JOB"
   --teacher-native-fp32-source-commit "$NATIVE_V2_FP32_SOURCE_COMMIT")
 fi
 run -m evals.context_training_proof "${ARGS[@]}" \
  --teacher-native-math-audit "$DL/runs/native-v2-native-approval-$NATIVE_V2_APPROVAL_JOB/report.json" \
  --out "$DL/runs/native-v2-native-math-$SLURM_JOB_ID" "${EXTRA[@]}"
else
 echo 'Unsupported native-v2 approval phase' >&2
 exit 1
fi
