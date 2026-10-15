#!/bin/bash
#SBATCH --job-name=dl-persona-policy
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-persona-policy-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
COMMON=(--model "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
 --out "$DL/runs/personaplex-policy-${SLURM_JOB_ID}")
STATES=(--states "$DL/runs/personaplex-states-7435849" --states "$DL/runs/personaplex-states-7435911")
if [ -n "${PERSONA_STATE_ROOT:-}" ]; then STATES=(--states "$PERSONA_STATE_ROOT"); fi
PAYLOAD=(apptainer exec --nv --bind /projects,/lus,/home
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/personaplex/moshi:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo"
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 "$DL/pytorch-new.sif" python -u)
"${PAYLOAD[@]}" -m evals.personaplex_policy_fit "${COMMON[@]}" \
 --parent "$DL/runs/personaplex-train-7435667" --data "$DL/runs/recovery-data-7407554" \
 "${STATES[@]}" --steps 512
"${PAYLOAD[@]}" -m evals.personaplex_train "${COMMON[@]}" \
 --cache "$DL/runs/recovery-mixed-codes-7411513" --voice "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --template "$DL/runs/personaplex-bridge-7428894" --proof "$DL/runs/personaplex-history-7434702" --reload
