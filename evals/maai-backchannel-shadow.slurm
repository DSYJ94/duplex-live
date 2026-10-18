#!/bin/bash
#SBATCH --job-name=dl-bcdet-shadow
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-bcdet-shadow-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
: "${MAAI_ASSETS_JOB:?Tensor inventory required}"
: "${MAAI_HEAD_JOB:?Strict head proof required}"
for JOB in "$MAAI_ASSETS_JOB" "$MAAI_HEAD_JOB"; do [[ "$JOB" =~ ^[0-9]+$ ]]; done
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
EXTRA=()
if [ -n "${MAAI_ATTENTION_JOB:-}" ]; then
 [[ "$MAAI_ATTENTION_JOB" =~ ^[0-9]+$ ]]
 : "${MAAI_REFERENCE_SHADOW_JOB:?Original shadow required}"
 [[ "$MAAI_REFERENCE_SHADOW_JOB" =~ ^[0-9]+$ ]]
 EXTRA=(--attention-proof "$DL/runs/maai-backchannel-attention-$MAAI_ATTENTION_JOB/report.json" --reference-shadow "$DL/runs/maai-backchannel-shadow-$MAAI_REFERENCE_SHADOW_JOB")
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" python -u -m evals.maai_backchannel_shadow \
 "${EXTRA[@]}" --inventory "$DL/runs/maai-backchannel-assets-$MAAI_ASSETS_JOB/report.json" \
 --head-approval "$DL/runs/maai-backchannel-head-$MAAI_HEAD_JOB/report.json" \
 --stt-assets "$DL/runs/kyutai-stt-assets-7510852/assets.json" \
 --mimi-approval "$DL/runs/kyutai-tts-mimi-load-7510722/report.json" \
 --source-run "$DL/runs/duplex-hybrid-probe-7511790" \
 --source-run "$DL/runs/duplex-hybrid-probe-7511791" \
 --out "$DL/runs/maai-backchannel-shadow-${SLURM_JOB_ID}"
