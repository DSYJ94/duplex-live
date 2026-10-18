#!/bin/bash
#SBATCH --job-name=dl-application-profile
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:25:00
#SBATCH --output=logs/dl-application-profile-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
unset OPENAI_API_KEY
EXTRA=()
if [ -n "${APPLICATION_CANDIDATE_JOB:-}" ]; then
 [[ "$APPLICATION_CANDIDATE_JOB" =~ ^[0-9]+$ ]]
 [[ "${APPLICATION_CANDIDATE_UPDATES:-}" =~ ^(972|1356)$ ]]
 EXTRA+=(--candidate "$DL/runs/controller-context-$APPLICATION_CANDIDATE_JOB" --candidate-updates "$APPLICATION_CANDIDATE_UPDATES")
 if [ "${APPLICATION_CANCEL_SCHEMA:-0}" = 1 ]; then EXTRA+=(--cancel-speech-contract); fi
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" python -u -m evals.application_capability_probe "${EXTRA[@]}" \
 --assets "$DL/runs/qwen35-controller-assets-7512612/assets.json" \
 --kernel-assets "$DL/runs/qwen35-kernel-assets-7513645" \
 --out "$DL/runs/application-capability-probe-${SLURM_JOB_ID}"
