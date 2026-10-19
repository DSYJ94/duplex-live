#!/bin/bash
#SBATCH --job-name=dl-native-fresh2-profile-probe
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-native-fresh2-profile-probe-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" python -u -m evals.native_fresh_source_profile_probe \
 --data "$DL/runs/native-fresh2-inputs-7549084" \
 --primary "$DL/runs/native-fresh2-judge-7549085/quality.json" \
 --diagnostic "$DL/runs/native-fresh2-asr-control-7549412/diagnostic.json" \
 --out "$DL/runs/native-fresh2-profile-probe-$SLURM_JOB_ID"
# Diagnostic only. This cannot approve input quality, training or native inference.
