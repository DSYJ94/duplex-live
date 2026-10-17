#!/bin/bash
#SBATCH --job-name=dl-cancel-ack-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-cancel-ack-cpu-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${CONTEXT_CANCEL_HISTORY_JOB:?Actual trained cancellation history required}"
: "${CONTEXT_ACTUAL_SOURCE_COMMIT:?Collector source required}"
[[ "$CONTEXT_CANCEL_HISTORY_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
OPTIONS=(--cancel-ack --teacher-id teacher-recovery-train-020)
if [ -n "${CONTEXT_CANCEL_CAPTURE_JOB:-}" ]; then
 [[ "$CONTEXT_CANCEL_CAPTURE_JOB" =~ ^[0-9]+$ ]]
 OPTIONS=(--audit-future-capture "$DL/runs/context-training-proof-$CONTEXT_CANCEL_CAPTURE_JOB")
fi
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 "$DL/pytorch-new.sif" \
 python -u -m evals.context_future_component --history "$DL/runs/context-spanning-probe-$CONTEXT_CANCEL_HISTORY_JOB" \
 --source-commit "$CONTEXT_ACTUAL_SOURCE_COMMIT" --request-policy native-prelead-epad \
 --data "$DL/runs/recovery-data-7407554" --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
 --tokenizer "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309/tokenizer_spm_32k_3.model" \
 --cache "$DL/runs/recovery-mixed-codes-7411513" --cache "$DL/runs/recovery-codes-7434926" \
 --cache "$DL/runs/recovery-codes-7435538" "${OPTIONS[@]}" \
 --out "$DL/runs/context-cancel-ack-eligibility-${SLURM_JOB_ID}"
