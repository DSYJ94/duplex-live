#!/bin/bash
#SBATCH --job-name=dl-native-persona-repair-asr
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-persona-repair-asr-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}" "${NATIVE_PERSONA_REPAIR_JOB:?}"
[[ "$NATIVE_PERSONA_REPAIR_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
RUNS=()
for I in $(seq 0 9); do
 JOB=7570828
 if [ "$I" = 4 ] || [ "$I" = 5 ]; then JOB="$NATIVE_PERSONA_REPAIR_JOB"; fi
 RUNS+=("$DL/runs/native-persona-diagnostic-${JOB}_$I")
done
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 --env HF_HOME="$DL/models/hf-cache" --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
 -m evals.moshi_speech_audit --runs "${RUNS[@]}" --device cpu \
 --trim-leading-silence --trim-trailing-silence --keep-silent-failures \
 --out "$DL/runs/native-persona-diagnostic-repair-asr-$SLURM_JOB_ID"
