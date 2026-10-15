#!/bin/bash
#SBATCH --job-name=dl-context-train-proof
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --output=logs/dl-context-train-proof-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
ARGS=(--assets "$DL/runs/context-spanning-assets-7454205/manifest.json"
 --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
 --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav"
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json"
 --cache "$DL/runs/recovery-mixed-codes-7411513"
 --cache "$DL/runs/recovery-codes-7434926" --cache "$DL/runs/recovery-codes-7435538"
 --out "$DL/runs/context-training-proof-${SLURM_JOB_ID}")
if [ "${CONTEXT_PILOT:-0}" = 1 ]; then
 ARGS+=(--steps 128 --pilot-preflight "$DL/runs/context-training-proof-7455157"
  --layout-audit "$DL/runs/context-training-proof-7455458/history-audit.json")
fi
run_container() {
 apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" python -u "$@"
}
if [ "${CONTEXT_CODEC_AUDIT:-0}" = 1 ]; then
 run_container -m evals.context_codec_audit \
  --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
  --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
  --cache "$DL/runs/recovery-mixed-codes-7411513" --out "$DL/runs/context-codec-audit-${SLURM_JOB_ID}"
 exit 0
fi
run_container -m unittest evals.harness.test_context_training_loss
if [ "${CONTEXT_TEACHER_AUDIT:-0}" = 1 ]; then
 run_container -m evals.context_training_proof "${ARGS[@]}" \
  --teacher-audit "$DL/runs/context-training-proof-7455157" \
  --teacher-audit "$DL/runs/context-training-proof-7455712"
 exit 0
fi
if [ "${CONTEXT_HISTORY_AUDIT:-0}" = 1 ]; then
 run_container -m evals.context_training_proof "${ARGS[@]}" --history-audit-only
 exit 0
fi
run_container -m evals.context_training_proof "${ARGS[@]}"
run_container -m evals.context_training_proof "${ARGS[@]}" --reload
