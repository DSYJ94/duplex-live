#!/bin/bash
#SBATCH --job-name=dl-native-fresh2-inputs
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-native-fresh2-inputs-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
: "${NATIVE_FRESH_PHASE:?generate or judge}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
ARGS=()
PY=python
ENV="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages"
if [ "$NATIVE_FRESH_PHASE" = generate ]; then
 set -a
 source "$DL/keys/openai.env"
 set +a
elif [ "$NATIVE_FRESH_PHASE" = judge ]; then
 : "${NATIVE_FRESH_INPUTS_JOB:?Complete20 generated input manifest required}"
 [[ "$NATIVE_FRESH_INPUTS_JOB" =~ ^[0-9]+$ ]]
 ARGS+=(--data "$DL/runs/native-fresh2-inputs-$NATIVE_FRESH_INPUTS_JOB")
 PY="$DL/envs/moshi-annotation-7379026/bin/python"
 ENV="$DL/envs/moshi-annotation-7379026/lib/python3.12/site-packages"
 export OPENAI_API_KEY=
else
 exit 1
fi
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$ENV:$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 --env HF_HOME="$DL/models/hf-cache" \
 "$DL/pytorch-new.sif" "$PY" -u -m evals.native_fresh_inputs_v2 "$NATIVE_FRESH_PHASE" "${ARGS[@]}" \
 --out "$DL/runs/native-fresh2-${NATIVE_FRESH_PHASE/generate/inputs}-$SLURM_JOB_ID"
