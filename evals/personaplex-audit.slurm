#!/bin/bash
#SBATCH --job-name=dl-persona-audit
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-persona-audit-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
: "${PERSONA_AUDIT_JOBS:?Native replay job IDs required}"
RUNS=()
for JOB in $PERSONA_AUDIT_JOBS; do
 [[ "$JOB" =~ ^[0-9]+$ ]]
 RUNS+=("$DL/runs/personaplex-native-$JOB")
done
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u -m evals.moshi_speech_audit \
 --runs "${RUNS[@]}" --trim-trailing-silence --keep-silent-failures \
 --out "$DL/runs/personaplex-audit-${SLURM_JOB_ID}"
