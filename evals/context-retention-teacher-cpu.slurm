#!/bin/bash
#SBATCH --job-name=dl-retention-teacher-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-retention-teacher-cpu-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
run() {
 apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 "$DL/pytorch-new.sif" python -u "$@"
}
run -m unittest evals.harness.test_context_retention_teacher evals.harness.test_context_training_state \
 evals.harness.test_context_teacher_inventory
if [ "${CONTEXT_OBSERVED_NATIVE_BOUNDARY:-0}" = 1 ]; then run -m unittest evals.harness.test_context_observed_teacher; fi
OPTIONS=()
if [ "${CONTEXT_ALL_APPROVED_TEACHERS:-0}" = 1 ]; then OPTIONS+=(--all-approved-teachers); fi
if [ "${CONTEXT_OBSERVED_NATIVE_BOUNDARY:-0}" = 1 ]; then OPTIONS+=(--observed-native-boundary); fi
run -m evals.context_retention_teacher \
 --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
 --tokenizer "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309/tokenizer_spm_32k_3.model" \
 --voice "$DL/runs/context-training-proof-7505957/voice.pt" \
 --cache "$DL/runs/recovery-mixed-codes-7411513" --cache "$DL/runs/recovery-codes-7434926" \
 --cache "$DL/runs/recovery-codes-7435538" "${OPTIONS[@]}" --out "$DL/runs/context-retention-teacher-audit-${SLURM_JOB_ID}"
