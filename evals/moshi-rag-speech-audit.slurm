#!/bin/bash
#SBATCH --job-name=dl-moshi-rag-audit
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-moshi-rag-audit-%j.out
set -euo pipefail
: "${RAG_SPEECH_JOB:?Set the completed speech comparison job ID}"
[[ "$RAG_SPEECH_JOB" =~ ^[0-9]+$ ]]
DL=/projects/u6si/duplex-live
cd "$DL/repo"
test -f "$DL/runs/moshi-rag-speech-${RAG_SPEECH_JOB}-base/report.json"
test -f "$DL/runs/moshi-rag-speech-${RAG_SPEECH_JOB}-delayed-reference/report.json"
git rev-parse HEAD
apptainer exec --nv --bind /projects,/lus,/home --env PYTHONPATH="$DL/repo" \
    "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
    -m evals.moshi_speech_audit \
    --runs "$DL/runs/moshi-rag-speech-${RAG_SPEECH_JOB}-base" \
           "$DL/runs/moshi-rag-speech-${RAG_SPEECH_JOB}-delayed-reference" \
    --out "$DL/runs/moshi-rag-speech-audit-${SLURM_JOB_ID}"
