#!/bin/bash
#SBATCH --job-name=dl-persona-states
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-persona-states-%A_%a.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
: "${SLURM_ARRAY_TASK_ID:?Bounded TRAIN scene index required}"
EXTRA=()
if [ -n "${PERSONA_INSTRUCTIONS:-}" ]; then EXTRA+=(--instructions "$PERSONA_INSTRUCTIONS"); fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/personaplex/moshi:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.personaplex_policy_collect \
 --source "$DL/personaplex" --data "$DL/runs/recovery-data-7407554" \
 --model "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
 --voice "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --index "$SLURM_ARRAY_TASK_ID" --out "$DL/runs/personaplex-states-${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}" "${EXTRA[@]}"
