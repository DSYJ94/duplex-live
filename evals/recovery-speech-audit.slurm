#!/bin/bash
# Bounded development speech audit payload for the exploratory reservation.
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
: "${RECOVERY_CANDIDATE_VOICE_JOB:?Candidate recording job ID required}"
[[ "$RECOVERY_CANDIDATE_VOICE_JOB" =~ ^[0-9]+$ ]]
EXTRA_RUNS=()
if [ -n "${RECOVERY_EXTRA_VOICE_JOBS:-}" ]; then
  read -r -a EXTRA_JOBS <<< "$RECOVERY_EXTRA_VOICE_JOBS"
  for JOB in "${EXTRA_JOBS[@]}"; do
    [[ "$JOB" =~ ^[0-9]+$ ]]
    EXTRA_RUNS+=("$DL/runs/recovery-voice-${JOB}/t1")
  done
fi
apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u \
  -m evals.moshi_speech_audit --trim-trailing-silence \
  --runs "$DL/runs/recovery-voice-7408188/t1" "$DL/runs/recovery-voice-${RECOVERY_CANDIDATE_VOICE_JOB}/t1" "${EXTRA_RUNS[@]}" \
  --live-runs "$DL/runs/gpt-live-fresh-7408224/t1" \
  --out "$DL/runs/recovery-speech-audit-${SLURM_JOB_ID}"
