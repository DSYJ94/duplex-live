#!/bin/bash
#SBATCH --job-name=dl-endpoint-replay-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/dl-endpoint-replay-cpu-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${KYUTAI_STT_PROBE_JOB:?Complete actual streaming ASR probe required}"
: "${KYUTAI_STT_ASSETS_JOB:?Pinned ASR assets required}"
[[ "$KYUTAI_STT_PROBE_JOB" =~ ^[0-9]+$ && "$KYUTAI_STT_ASSETS_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
run_container() {
 apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
 "$DL/pytorch-new.sif" python -u "$@"
}
run_container -m unittest evals.harness.test_duplex_endpoint
run_container -m evals.duplex_endpoint_replay \
 --asr-run "$DL/runs/kyutai-stt-probe-$KYUTAI_STT_PROBE_JOB" \
 --assets "$DL/runs/kyutai-stt-assets-$KYUTAI_STT_ASSETS_JOB/assets.json" \
 --out "$DL/runs/duplex-endpoint-replay-${SLURM_JOB_ID}"
