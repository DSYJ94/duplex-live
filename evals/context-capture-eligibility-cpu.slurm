#!/bin/bash
#SBATCH --job-name=dl-control-eligibility
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-control-eligibility-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${DL_CAPTURE_MAP:?Recorded conditioning map required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
REGRESSIONS=()
OPTIONS=()
if [ "${CONTEXT_ORDINARY_RESPONSE_BOUNDARY:-0}" = 1 ]; then OPTIONS+=(--ordinary-response-boundary); fi
if [ "${CONTEXT_LATE_DIAGNOSTICS:-0}" = 1 ]; then OPTIONS+=(--late-diagnostics); fi
for J in ${CONTEXT_REGRESSION_FUTURES:-7467981 7468102 7468611 7479546}; do
 [[ "$J" =~ ^[0-9]+$ ]]
 REGRESSIONS+=(--regression-capture "$DL/runs/context-training-proof-$J")
done
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 "$DL/pytorch-new.sif" \
 python -u -m evals.context_capture_eligibility --map "$DL_CAPTURE_MAP" \
 --data "$DL/runs/recovery-data-7407554" --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
 --tokenizer "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309/tokenizer_spm_32k_3.model" \
 --cache "$DL/runs/recovery-mixed-codes-7411513" --cache "$DL/runs/recovery-codes-7434926" \
 --cache "$DL/runs/recovery-codes-7435538" "${REGRESSIONS[@]}" \
 "${OPTIONS[@]}" --out "$DL/runs/context-control-eligibility-${SLURM_JOB_ID}"
