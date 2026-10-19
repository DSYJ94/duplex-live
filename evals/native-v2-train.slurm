#!/bin/bash
#SBATCH --job-name=dl-native-v2-train-proof
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-native-v2-train-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable committed source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
for NAME in NATIVE_V2_CODES_JOB NATIVE_V2_SCOPE_JOB NATIVE_V2_APPROVAL_JOB NATIVE_V2_MATH_JOB \
 NATIVE_OLD25_CODES_JOB NATIVE_OLD25_SCOPE_JOB NATIVE_OLD25_APPROVAL_JOB NATIVE_OLD25_MATH_JOB; do
 [[ "${!NAME:?Complete independent native-v2 approvals required}" =~ ^[0-9]+$ ]]
done
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
MOSHI="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
test "$(git -C "$MOSHI" rev-parse HEAD)" = 8c6dfc101b7871baa428424bcdc583b74fb561d9
PARENT="$DL/runs/context-training-proof-7510331"
PHASE="${NATIVE_V2_TRAIN_PHASE:-train}"
case "$PHASE" in
 train) RUN_OUT="$DL/runs/context-training-proof-$SLURM_JOB_ID" ;;
 gradient-audit) RUN_OUT="$DL/runs/native-v2-gradient-audit-$SLURM_JOB_ID" ;;
 *) echo "Unknown native phase" >&2; exit 2 ;;
esac
ARGS=(--assets "$DL/runs/context-spanning-assets-7454205/manifest.json"
 --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
 --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav"
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json"
 --cache "$DL/runs/recovery-mixed-codes-7411513"
 --cache "$DL/runs/recovery-codes-7434926" --cache "$DL/runs/recovery-codes-7435538"
 --out "$RUN_OUT"
 --request-policy native-prelead-epad --actual-source-data "$DL/runs/recovery-data-7407554"
 --actual-temporal-fit --warm-start-proof "$PARENT" --actual-fit-updates 8
 --actual-sampling balanced-kind --actual-retention-sampling balanced-kind
 --actual-base-retention-weight 1 --actual-retention-teacher-weight 1
 --actual-native-teacher-loss balanced-spoken --actual-response-first-word-weight 16
 --actual-native-teacher-coverage all-approved
 --actual-native-teacher-cache "$DL/runs/native-old25-codes-$NATIVE_OLD25_CODES_JOB"
 --actual-native-teacher-approval "$DL/runs/native-old25-native-approval-$NATIVE_OLD25_APPROVAL_JOB/report.json"
 --actual-native-teacher-scope "$DL/runs/native-old25-native-scope-$NATIVE_OLD25_SCOPE_JOB/report.json"
 --actual-native-teacher-preflight "$DL/runs/native-old25-native-math-$NATIVE_OLD25_MATH_JOB"
 --native-v2-cache "$DL/runs/native-v2-codes-$NATIVE_V2_CODES_JOB"
 --native-v2-scope "$DL/runs/native-v2-native-scope-$NATIVE_V2_SCOPE_JOB/report.json"
 --native-v2-approval "$DL/runs/native-v2-native-approval-$NATIVE_V2_APPROVAL_JOB/report.json"
 --native-v2-preflight "$DL/runs/native-v2-native-math-$NATIVE_V2_MATH_JOB")
mapfile -t COMPONENTS < <(python3 -c 'import json,sys,hashlib; from pathlib import Path; p=Path(sys.argv[1]); r=json.loads((p/"report.json").read_text()); d=(p/"data.json").read_bytes(); assert hashlib.sha256(d).hexdigest()==r["data_sha256"]; rows=json.loads(d)["components"]; assert len(rows)==28; print("\n".join(x["component"] for x in rows))' "$PARENT")
test "${#COMPONENTS[@]}" = 28
for COMPONENT in "${COMPONENTS[@]}"; do ARGS+=(--actual-mixed-data "$COMPONENT"); done
PREFLIGHT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["actual_temporal_preflight"])' "$PARENT/report.json")
ARGS+=(--actual-temporal-preflight "$PREFLIGHT")
run() {
 apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$MOSHI/moshi:$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env OPENAI_API_KEY= \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" python -u "$@"
}
run -m unittest evals.harness.test_context_training_state evals.harness.test_context_native_v2_teacher_objective
if [[ "$PHASE" == gradient-audit ]]; then
 run -m evals.context_training_proof "${ARGS[@]}" --native-v2-gradient-audit
 test -f "$RUN_OUT/native-v2-gradient-audit.json"
 test ! -e "$RUN_OUT/adapters.safetensors"
 test ! -e "$RUN_OUT/training-state.pt"
 exit 0
fi
run -m evals.context_training_proof "${ARGS[@]}"
# This is a fresh process, with complete adapter/optimizer/Python/NumPy/Torch/CUDA restoration.
run -m evals.context_training_proof "${ARGS[@]}" --reload
# No extension or adoption is authorized by this mechanics result alone.
