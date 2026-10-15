#!/bin/bash
#SBATCH --job-name=dl-lychee-fixed-state
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-lychee-fixed-state-%j.out
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
for J in 7452624 7452625 7452626 7452627 7452628 7452629 7452630 7452631 7452946 7452947 7452948 7452949 7452950 7452951 7452952 7452953 7452954 7452955 7452956 7452957 7452958; do
 STATES+=("$DL/runs/lychee-vllm-probe-$J")
done
EXTRA=()
if [ -n "${LYCHEE_SEMANTIC_JOB:-}" ]; then
 [[ "$LYCHEE_SEMANTIC_JOB" =~ ^[0-9]+$ ]]
 EXTRA+=(--semantic-adapter "$DL/runs/lychee-semantic-proof-$LYCHEE_SEMANTIC_JOB/adapter.pt"
  --semantic-proof "$DL/runs/lychee-semantic-proof-$LYCHEE_SEMANTIC_JOB/report.json")
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$RUNTIME:$DEPS:$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.lychee_vllm_probe \
 --assets "$DL/runs/lychee-assets-7414609/manifest.json" --batch "$DL/runs/synth-scale1-q/audio" \
 --runtime "$RUNTIME" --preflight "$DL/runs/lychee-vllm-preflight-7441739" --warm-input \
 --fixed-state-data "$DL/runs/lychee-training-inputs-7453119" --fixed-state-histories "${STATES[@]}" \
 --fixed-state-scenarios recovery-train-000 recovery-train-001 teacher-cf-recovery-train-000-1 \
 --out "$DL/runs/lychee-fixed-state-${SLURM_JOB_ID}" "${EXTRA[@]}"
