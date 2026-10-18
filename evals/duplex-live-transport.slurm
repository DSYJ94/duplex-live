#!/bin/bash
#SBATCH --job-name=dl-live-transport-loopback
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-live-transport-loopback-%j.out
set -euo pipefail
export DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
export DL_LIVE_INPUTS_JOB=7515325 DL_LIVE_QUALITY_JOB=7515326 DL_LIVE_VALIDATOR=c10fddf DL_LIVE_VALIDATOR_COMMIT=c10fddffbf7ea47b7c3fe628b1cd0d01d2f06156 DL_LIVE_SCENARIO=hybrid-fresh-024
if [ "${DUPLEX_LIVE_REASONING_CANCEL:-0}" = 1 ]; then
 export DL_LIVE_INPUTS_JOB=7515617 DL_LIVE_QUALITY_JOB=7515618 DL_LIVE_VALIDATOR=42c704d DL_LIVE_VALIDATOR_COMMIT=42c704d7ef3472eb71a701a5db6a2dfee7c9ca69 DL_LIVE_SCENARIO=hybrid-fresh-026
fi
export DL_LIVE_RUN="$DL/runs/duplex-live-transport-$SLURM_JOB_ID"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" bash -c '
 set -euo pipefail
 python -u -m serving.hybrid.server \
 --stt-assets "$DL/runs/kyutai-stt-assets-7510852/assets.json" \
 --tts-assets "$DL/runs/kyutai-tts-assets-7510636/assets.json" \
 --mimi-load-approval "$DL/runs/kyutai-tts-mimi-load-7510722/report.json" \
 --controller-assets "$DL/runs/qwen35-controller-assets-7512612/assets.json" \
 --kernel-assets "$DL/runs/qwen35-kernel-assets-7513645" \
 --out "$DL_LIVE_RUN" --port 8788 --max-sessions 1 --idle-seconds 300 &
 service_pid=$!
 trap "kill $service_pid 2>/dev/null || true" EXIT
 python -u -m serving.hybrid.loopback --server "$DL_LIVE_RUN" \
 --data "$DL/runs/duplex-fresh-inputs-$DL_LIVE_INPUTS_JOB" \
 --quality "$DL/runs/duplex-fresh-quality-$DL_LIVE_QUALITY_JOB/quality.json" \
 --validator-source "$DL/worktrees/recovery-$DL_LIVE_VALIDATOR" \
 --validator-commit "$DL_LIVE_VALIDATOR_COMMIT" \
 --scenario "$DL_LIVE_SCENARIO" --out "$DL_LIVE_RUN-loopback"
 wait "$service_pid"
 '
