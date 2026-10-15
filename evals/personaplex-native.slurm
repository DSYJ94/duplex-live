#!/bin/bash
#SBATCH --job-name=dl-persona-native
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-persona-native-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
: "${PERSONA_CHECKPOINT:?Verified native joint checkpoint required}"
: "${PERSONA_SCENARIO:?Approved validation scenario required}"
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/personaplex/moshi:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.personaplex_native_probe \
 --source "$DL/personaplex" \
 --model "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
 --voice "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --data "$DL/runs/recovery-data-7407554" --scenario "$PERSONA_SCENARIO" \
 --checkpoint "$PERSONA_CHECKPOINT" \
 --arc-source "$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9" \
 --manifest "$DL/runs/moshi-models-7378305/model-manifest.json" \
 --arc-tokenizer "$DL/models/hf-cache/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
 --out "$DL/runs/personaplex-native-${SLURM_JOB_ID}"
