#!/bin/bash
#SBATCH --job-name=dl-control-rollin-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-control-rollin-cpu-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${CONTEXT_CONTROL_HISTORY_JOB:?Actual trained TRAIN history required}"
: "${CONTEXT_CONTROL_KIND:?Explicit control kind required}"
: "${CONTEXT_CONTROL_TEACHER_ID:?Approved independent teacher required}"
[[ "$CONTEXT_CONTROL_HISTORY_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
EXTRA=()
if [ -n "${CONTEXT_CONTROL_CAPTURE_JOB:-}" ]; then
 [[ "$CONTEXT_CONTROL_CAPTURE_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--audit-capture "$DL/runs/context-training-proof-$CONTEXT_CONTROL_CAPTURE_JOB")
fi
cd "$DL_TASK_SOURCE"
run() {
 apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 "$DL/pytorch-new.sif" python -u "$@"
}
run -m unittest evals.harness.test_context_control_rollin evals.harness.test_context_component_kind evals.harness.test_context_training_state
run -m evals.context_control_rollin --history "$DL/runs/context-spanning-probe-$CONTEXT_CONTROL_HISTORY_JOB" \
 --kind "$CONTEXT_CONTROL_KIND" --teacher-id "$CONTEXT_CONTROL_TEACHER_ID" \
 --data "$DL/runs/recovery-data-7407554" --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
 --tokenizer "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309/tokenizer_spm_32k_3.model" \
 --cache "$DL/runs/recovery-mixed-codes-7411513" --cache "$DL/runs/recovery-codes-7434926" \
 --cache "$DL/runs/recovery-codes-7435538" \
 --regression-capture "$DL/runs/context-training-proof-7505218" \
 --regression-capture "$DL/runs/context-training-proof-7495655" \
 --regression-capture "$DL/runs/context-training-proof-7504669" \
 --regression-capture "$DL/runs/context-training-proof-7501220" \
 "${EXTRA[@]}" --out "$DL/runs/context-control-rollin-eligibility-${SLURM_JOB_ID}"
