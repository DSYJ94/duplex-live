#!/bin/bash
#SBATCH --job-name=dl-lychee-train-inputs
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-lychee-train-inputs-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
RUNTIME="$DL/envs/lychee-vllm-7441739"
DEPS="$DL/envs/lychee-vllm-deps-7441737"
STATES=()
for J in 7452624 7452625 7452626 7452627 7452628 7452629 7452630 7452631; do
 STATES+=("$DL/runs/lychee-vllm-probe-$J")
done
EXTRA=()
if [ "${LYCHEE_PAIRED_HISTORIES:-0}" = 1 ]; then
 for J in 7452946 7452947 7452948 7452949 7452950 7452951 7452952 7452953 7452954 7452955 7452956 7452957 7452958; do
  STATES+=("$DL/runs/lychee-vllm-probe-$J")
 done
 EXTRA+=(--training-source-data "$DL/runs/recovery-teacher-data-7434749")
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$RUNTIME:$DEPS:$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.lychee_vllm_probe \
 --assets "$DL/runs/lychee-assets-7414609/manifest.json" \
 --batch "$DL/runs/synth-scale1-q/audio" --runtime "$RUNTIME" \
 --preflight "$DL/runs/lychee-vllm-preflight-7441739" --warm-input \
 --data "$DL/runs/recovery-data-7407554" --training-states "${STATES[@]}" "${EXTRA[@]}" \
 --codec-data "$DL/runs/lychee-codec-data-7452500" --codec-audit "$DL/runs/lychee-codec-audit-7452502" \
 --out "$DL/runs/lychee-training-inputs-${SLURM_JOB_ID}"
