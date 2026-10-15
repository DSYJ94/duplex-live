#!/bin/bash
#SBATCH --job-name=dl-lychee-vocode
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-lychee-vocode-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
TASK_SOURCE="${DL_TASK_SOURCE:-$DL/repo}"
if [ "$TASK_SOURCE" != "$DL/repo" ]; then
 : "${DL_TASK_COMMIT:?Immutable source commit required}"
 test "$(git -C "$TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
 test -z "$(git -C "$TASK_SOURCE" status --porcelain)"
fi
cd "$TASK_SOURCE"
: "${LYCHEE_NATIVE_JOB:?Strict full-model native token run required}"
[[ "$LYCHEE_NATIVE_JOB" =~ ^[0-9]+$ ]]
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.lychee_vocode \
 --assets "$DL/runs/lychee-assets-7414609/manifest.json" \
 --native-run "$DL/runs/lychee-vllm-probe-$LYCHEE_NATIVE_JOB" \
 --voice "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json" \
 --mode "${LYCHEE_VOCODE_MODE:-offline}" --hop "${LYCHEE_VOCODE_HOP:-25}" \
 --out "$DL/runs/lychee-vocode-${SLURM_JOB_ID}"
