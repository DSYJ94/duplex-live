#!/bin/bash
#SBATCH --job-name=dl-fresh-hybrid-inputs
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-fresh-hybrid-inputs-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
EXTRA=()
if [ "${DUPLEX_FRESH_REVOICE:-0}" = 1 ]; then
 EXTRA=(--case hybrid-fresh-005 --case hybrid-fresh-006 --user-voice coral --backchannel-wording uh-huh)
fi
if [ "${DUPLEX_FRESH_NEW_CORRECTION:-0}" = 1 ]; then
 EXTRA=(--case hybrid-fresh-007 --user-voice coral)
fi
if [ "${DUPLEX_FRESH_SECOND_BATCH:-0}" = 1 ]; then
 EXTRA=(--user-voice coral)
 for ID in 008 009 010 011 012 013 014 015; do EXTRA+=(--case "hybrid-fresh-$ID"); done
fi
if [ "${DUPLEX_FRESH_THIRD_BATCH:-0}" = 1 ]; then
 EXTRA=(--user-voice coral)
 for ID in 016 017 018 019 020 021 022 023; do EXTRA+=(--case "hybrid-fresh-$ID"); done
fi
if [ "${DUPLEX_FRESH_REASONING_BATCH:-0}" = 1 ]; then
 EXTRA=(--case hybrid-fresh-024 --case hybrid-fresh-025 --user-voice coral)
fi
if [ "${DUPLEX_FRESH_REASONING_CANCEL:-0}" = 1 ]; then
 EXTRA=(--case hybrid-fresh-026 --user-voice coral)
fi
if [ "${DUPLEX_FRESH_STATE_CONSTRAINTS:-0}" = 1 ]; then
 EXTRA=(--case hybrid-fresh-027 --case hybrid-fresh-028 --user-voice coral)
fi
if [ "${DUPLEX_FRESH_SINGLE_STOP:-0}" = 1 ]; then
 EXTRA=(--case hybrid-fresh-029 --user-voice coral)
fi
if [ "${DUPLEX_FRESH_SUMMARY_REVISION:-0}" = 1 ]; then
 EXTRA=(--case hybrid-fresh-030 --user-voice coral)
fi
if [ "${DUPLEX_FRESH_SPEECH_HOLDOUT:-0}" = 1 ]; then
 EXTRA=(--user-voice coral)
 for ID in 031 032 033 034; do EXTRA+=(--case "hybrid-fresh-$ID"); done
fi
if [ "${DUPLEX_FRESH_CONTROL_QUALIFICATION:-0}" = 1 ]; then
 EXTRA=(--user-voice coral)
 for ID in 035 036 037 038 039 040 041; do EXTRA+=(--case "hybrid-fresh-$ID"); done
fi
if [ "${DUPLEX_FRESH_SEMANTIC_RESUME:-0}" = 1 ]; then
 EXTRA=()
 for ID in 042 043 044 045; do EXTRA+=(--case "hybrid-fresh-$ID"); done
fi
if [ "${DUPLEX_CANCEL_RESUME_INPUT:-0}" = 1 ]; then
 EXTRA=(--case hybrid-fresh-046 --user-voice coral)
fi
if [ "${DUPLEX_FROZEN_CONVERSATION_EVAL:-0}" = 1 ]; then
 EXTRA=(--user-voice coral)
 for ID in 047 048 049 050 051 052 053 054 055 056 057; do EXTRA+=(--case "hybrid-fresh-$ID"); done
fi
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
 "$DL/pytorch-new.sif" python -u -m evals.duplex_fresh_inputs generate "${EXTRA[@]}" \
 --out "$DL/runs/duplex-fresh-inputs-${SLURM_JOB_ID}"
