#!/bin/bash
#SBATCH --job-name=dl-native-conditioning
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:25:00
#SBATCH --output=logs/dl-native-conditioning-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}" "${DL_TASK_COMMIT:?}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
OUT="$DL/runs/native-conditioning-$SLURM_JOB_ID"
mkdir "$OUT"
scontrol show job "$SLURM_JOB_ID" > "$OUT/allocation.txt"
run_probe() {
 apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env DL_TASK_COMMIT="$DL_TASK_COMMIT" --env HF_HOME="$DL/models/hf-cache" \
 --env HF_HUB_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" python -u -m evals.native_conditioning_probe "$@"
}
run_probe --phase capture --capture "$OUT/capture" --out "$OUT/capture"
for CHECKPOINT in 7510331 7558824; do
 run_probe --phase score --checkpoint "$CHECKPOINT" --capture "$OUT/capture" --out "$OUT/score-$CHECKPOINT"
done
# Zero optimizer/API calls. Forced teacher text; never native reasoning approval.
