#!/bin/bash
set -euo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1
DL=/projects/u6si/duplex-live
test "${NATIVE_FRESH_SOURCE_PROFILE:-owner-approved-turbo-spelling-v1}" = owner-approved-turbo-spelling-v1
INDEX="${1:?Complete40 case index required}"
[[ "$INDEX" =~ ^[0-9]+$ && "$INDEX" -lt 40 ]]
: "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}" "${NATIVE_FRESH_CANDIDATE_JOB:?}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
readarray -t FIELDS < <(python3 -m evals.native_fresh_session launch \
 --candidate "$NATIVE_FRESH_CANDIDATE_JOB" --index "$INDEX" --allocation "$SLURM_JOB_ID" \
 --step "$SLURM_STEP_ID" --devices "$CUDA_VISIBLE_DEVICES" \
 --out "$DL/runs/native-fresh2-session-$SLURM_JOB_ID/case-$INDEX-launch.json")
test "${#FIELDS[@]}" = 5
EXTRA=()
QUALITY="$DL/runs/native-fresh2-source-reapproval-$NATIVE_FRESH_QUALITY_JOB/quality.json"
EXTRA+=(--native-fresh-source-profile owner-approved-turbo-spelling-v1)
if [ "${FIELDS[2]}" = native-asr ]; then
 EXTRA+=(--asr-python "$DL/envs/moshi-annotation-7379026/bin/python")
 set -a
 source "$DL/keys/openai.env"
 set +a
else
 export OPENAI_API_KEY=
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.context_spanning_fresh_probe \
 --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
 --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
 --data "$DL/runs/native-fresh2-inputs-$NATIVE_FRESH_INPUTS_JOB" \
 --native-fresh-candidate-proof "$DL/runs/context-training-proof-$NATIVE_FRESH_CANDIDATE_JOB" \
 --native-fresh-behavior-review "$NATIVE_FRESH_BEHAVIOR_REVIEW" \
 --native-fresh-quality "$QUALITY" \
 --scenario "${FIELDS[0]}" --expected-kind "${FIELDS[1]}" --backend-mode "${FIELDS[2]}" \
 --lookup-delivery-delay-s "${FIELDS[3]}" --adapter-proof "$DL/runs/context-training-proof-${FIELDS[4]}" \
 --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json" \
 --out "$DL/runs/native-fresh2-evaluate-${SLURM_JOB_ID}_$INDEX" "${EXTRA[@]}"
# Complete40 actual comparisons and independent speech/behavior/causality review are still required.
