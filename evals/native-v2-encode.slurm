#!/bin/bash
#SBATCH --job-name=dl-native-v2-encode
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-v2-encode-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable committed source required}"
: "${DL_TASK_COMMIT:?Immutable source identity required}"
: "${NATIVE_CONVERSION_JOB:?Complete semantic/conversion approval required}"
[[ "$NATIVE_CONVERSION_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
SOURCE="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 8c6dfc101b7871baa428424bcdc583b74fb561d9
cd "$DL_TASK_SOURCE"
export DL
apptainer exec --nv --bind /projects,/lus,/home \
 --env DL="$DL" --env DL_TASK_SOURCE="$DL_TASK_SOURCE" \
 --env NATIVE_CONVERSION_JOB="$NATIVE_CONVERSION_JOB" \
 --env SLURM_JOB_ID="$SLURM_JOB_ID" \
 --env PYTHONPATH="$SOURCE/moshi:$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
 --env TRANSFORMERS_OFFLINE=1 --env OMP_NUM_THREADS=8 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
PY="$DL/envs/moshi-rag-7379661/bin/python"
DATA="$DL/runs/native-v2-teacher-data-$NATIVE_CONVERSION_JOB"
# Revalidate the complete semantic/raw/quality chain before native codec execution.
"$PY" -c 'import sys; from pathlib import Path; from evals.recovery_data import validate_conversion; r=validate_conversion(Path(sys.argv[1])); assert r["schema"]=="dl-teacher-native-v1" and len(r["accepted"])==24 and r.get("semantic_approval")' "$DATA"
export TRITON_LIBCUDA_PATH
TRITON_LIBCUDA_PATH=$("$PY" -m evals.cuda_linker_path --out "$DL/runs/native-v2-encode-driver-$SLURM_JOB_ID")
"$PY" -u -m evals.moshi_reference_probe \
 --manifest "$DL/runs/moshi-models-7378305/model-manifest.json" \
 --tokenizer "$DL/models/hf-cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
 --out "$DL/runs/native-v2-encoder-$SLURM_JOB_ID" \
 --encode-data "$DATA" --encoded-out "$DL/runs/native-v2-codes-$SLURM_JOB_ID" --in-process-encode
CONTAINER
