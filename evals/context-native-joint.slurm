#!/bin/bash
#SBATCH --job-name=dl-native-joint
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-joint-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable committed source required}"
: "${DL_TASK_COMMIT:?Immutable source identity required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
PHASE=${NATIVE_JOINT_PHASE:-audit}
[[ "$PHASE" = audit || "$PHASE" = train ]]
EXTRA=()
if [ "$PHASE" = train ]; then
 : "${NATIVE_JOINT_AUDIT_JOB:?Independent complete audio-target/numerical audit required}"
 [[ "$NATIVE_JOINT_AUDIT_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--audit "$DL/runs/native-assistant-joint-$NATIVE_JOINT_AUDIT_JOB")
fi
ARGS=(--assets "$DL/runs/context-spanning-assets-7454205/manifest.json"
 --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
 --parent "$DL/runs/context-training-proof-7510331"
 --approval "$DL/runs/context-teacher-native-approval-7508795/report.json"
 --scope "$DL/runs/context-retention-teacher-audit-7508779/report.json"
 --numerical "$DL/runs/context-training-proof-7509062"
 --cache "$DL/runs/recovery-mixed-codes-7411513"
 --cache "$DL/runs/recovery-codes-7434926" --cache "$DL/runs/recovery-codes-7435538"
 --out "$DL/runs/native-assistant-joint-${SLURM_JOB_ID}" "${EXTRA[@]}")
run_native() {
 apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 \
  --env OPENAI_API_KEY= \
  "$DL/pytorch-new.sif" python -u -m evals.context_native_joint "${ARGS[@]}" --phase "$1"
}
run_native "$PHASE"
if [ "$PHASE" = train ]; then run_native restore; fi
