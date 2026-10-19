#!/bin/bash
#SBATCH --job-name=dl-native-old25-native-approval
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --output=logs/dl-native-old25-native-%A_%a.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable committed source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
: "${NATIVE_OLD25_CODES_JOB:?Complete native-old25 codec cache required}"
: "${NATIVE_OLD25_PHASE:?Choose scope, capture, approve, or math}"
[[ "$NATIVE_OLD25_CODES_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
CACHE="$DL/runs/native-old25-codes-$NATIVE_OLD25_CODES_JOB"
ASSETS="$DL/runs/context-spanning-assets-7454205/manifest.json"
BASE="$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
VOICE="$DL/runs/context-training-proof-7510331/voice.pt"
MOSHI="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
test "$(git -C "$MOSHI" rev-parse HEAD)" = 8c6dfc101b7871baa428424bcdc583b74fb561d9
NV=(--nv)
if [ "${NATIVE_OLD25_DEVICE:-cuda}" = cpu ]; then
 [[ "$NATIVE_OLD25_PHASE" = scope || "$NATIVE_OLD25_PHASE" = approve ]]
 NV=()
fi
run() {
 apptainer exec "${NV[@]}" --bind /projects,/lus,/home \
  --env PYTHONPATH="$MOSHI/moshi:$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env OPENAI_API_KEY= \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" python -u "$@"
}
# All25 original histories; only the explicitly approved four derived timings change.
run -c 'import sys; from pathlib import Path; from evals.recovery_data import validate_cache; from evals.native_old25_timing_repair import inventory; r=validate_cache(Path(sys.argv[1])); assert r["schema"]=="dl-native-old25-timing-cache-v1" and len(r["examples"])==25 and {e["scenario_id"] for e in r["examples"]}=={e["teacher"]["id"] for e in inventory()} and all(e["split"]=="train" for e in r["examples"])' "$CACHE"
if [ "$NATIVE_OLD25_PHASE" = scope ]; then
 run -m evals.context_retention_teacher --assets "$ASSETS" --tokenizer "$BASE/tokenizer_spm_32k_3.model" \
  --voice "$VOICE" --cache "$CACHE" --all-approved-teachers --observed-native-boundary \
  --out "$DL/runs/native-old25-native-scope-$SLURM_JOB_ID"
 run -c 'import copy,json,sys; from evals.context_native_v2_teacher_objective import validate_old25_lineage; p=json.load(open("/projects/u6si/duplex-live/runs/context-training-proof-7510331/report.json"))["retention_teacher_text"]; r=copy.deepcopy(p); r["teachers"]=json.load(open(sys.argv[1]))["teachers"]; validate_old25_lineage(p,r); print("All25 scope preserves words/audio/frames/masks and all21 undiagnosed targets exactly")' \
  "$DL/runs/native-old25-native-scope-$SLURM_JOB_ID/report.json"
 exit 0
fi
: "${NATIVE_OLD25_SCOPE_JOB:?Independent complete native-old25 scope required}"
[[ "$NATIVE_OLD25_SCOPE_JOB" =~ ^[0-9]+$ ]]
SCOPE="$DL/runs/native-old25-native-scope-$NATIVE_OLD25_SCOPE_JOB/report.json"
ARGS=(--assets "$ASSETS" --base "$BASE"
 --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav"
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json"
 --cache "$CACHE" --request-policy native-prelead-epad
 --teacher-engine-input-audit "$SCOPE" --teacher-engine-input-voice "$VOICE")
if [ "$NATIVE_OLD25_PHASE" = capture ]; then
 : "${SLURM_ARRAY_JOB_ID:?Submit the complete 0-24 array with at most four concurrent tasks}"
 [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ && "$SLURM_ARRAY_TASK_ID" -lt 25 ]]
 ID=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["passed"] and len(r["teachers"])==25; print(r["teachers"][int(sys.argv[2])]["id"])' "$SCOPE" "$SLURM_ARRAY_TASK_ID")
 run -m evals.context_training_proof "${ARGS[@]}" --teacher-engine-input-id "$ID" \
  --out "$DL/runs/native-old25-native-capture-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID"
elif [ "$NATIVE_OLD25_PHASE" = approve ]; then
 : "${NATIVE_OLD25_CAPTURE_ARRAY_JOB:?Complete native teacher trace array required}"
 [[ "$NATIVE_OLD25_CAPTURE_ARRAY_JOB" =~ ^[0-9]+$ ]]
 CAPTURES=()
 for I in $(seq 0 24); do
  CAPTURES+=(--capture "$DL/runs/native-old25-native-capture-${NATIVE_OLD25_CAPTURE_ARRAY_JOB}_$I")
 done
 run -m evals.context_teacher_native_approval --assets "$ASSETS" \
  --tokenizer "$BASE/tokenizer_spm_32k_3.model" --voice "$VOICE" --cache "$CACHE" \
  --scope "$SCOPE" "${CAPTURES[@]}" --out "$DL/runs/native-old25-native-approval-$SLURM_JOB_ID"
elif [ "$NATIVE_OLD25_PHASE" = math ]; then
 : "${NATIVE_OLD25_APPROVAL_JOB:?Independent complete input approval required}"
 [[ "$NATIVE_OLD25_APPROVAL_JOB" =~ ^[0-9]+$ ]]
 EXTRA=()
 if [ -n "${NATIVE_OLD25_FP32_PREFLIGHT_JOB:-}" ]; then
  : "${NATIVE_OLD25_FP32_SOURCE_COMMIT:?Original FP32 audit source required}"
  [[ "$NATIVE_OLD25_FP32_PREFLIGHT_JOB" =~ ^[0-9]+$ ]]
  EXTRA+=(--teacher-native-fp32-preflight "$DL/runs/native-old25-native-math-$NATIVE_OLD25_FP32_PREFLIGHT_JOB"
   --teacher-native-fp32-source-commit "$NATIVE_OLD25_FP32_SOURCE_COMMIT")
 fi
 run -m evals.context_training_proof "${ARGS[@]}" \
  --teacher-native-math-audit "$DL/runs/native-old25-native-approval-$NATIVE_OLD25_APPROVAL_JOB/report.json" \
  --out "$DL/runs/native-old25-native-math-$SLURM_JOB_ID" "${EXTRA[@]}"
else
 echo 'Unsupported native-old25 approval phase' >&2
 exit 1
fi
