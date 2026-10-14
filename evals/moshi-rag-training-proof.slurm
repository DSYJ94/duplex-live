#!/bin/bash
#SBATCH --job-name=dl-rag-training-proof
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-rag-training-proof-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
SOURCE="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 8c6dfc101b7871baa428424bcdc583b74fb561d9
export DL SOURCE
export RAG_PILOT_STEPS="${RAG_PILOT_STEPS:-0}"
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$SOURCE/moshi:$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
PY="$DL/envs/moshi-rag-7379661/bin/python"
export TRITON_LIBCUDA_PATH
TRITON_LIBCUDA_PATH=$("$PY" -m evals.cuda_linker_path --out "$DL/runs/rag-training-driver-${SLURM_JOB_ID}")
"$PY" -u -m evals.moshi_reference_probe \
  --manifest "$DL/runs/moshi-models-7378305/model-manifest.json" \
  --tokenizer "$DL/models/hf-cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
  --out "$DL/runs/rag-training-reference-${SLURM_JOB_ID}" \
  --training-proof "$DL/runs/rag-training-proof-${SLURM_JOB_ID}" \
  --training-steps "$RAG_PILOT_STEPS"
CONTAINER
