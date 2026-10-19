#!/bin/bash
#SBATCH --job-name=dl-interactive-probe
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-interactive-probe-%j.out
# No partition or QoS override: the site's reservation selects both.
set -euo pipefail
cd /projects/u6si/duplex-live/repo
case "${1:?Choose rag, personaplex or qwen}" in
  confirmed-pause-network)
    : "${DL_TASK_SOURCE:?Immutable source required}"
    : "${DL_TASK_COMMIT:?Immutable commit required}"
    test "${DUPLEX_CONFIRMED_PAUSE:-0}" = 1
    test "${DUPLEX_LIVE_SCENARIO:-}" = hybrid-fresh-051
    SCRIPT="$DL_TASK_SOURCE/evals/duplex-live-transport.slurm" ;;
  native-v2-gradient-audit)
    : "${DL_TASK_SOURCE:?Immutable diagnostic payload source required}"
    : "${DL_TASK_COMMIT:?Immutable diagnostic payload commit required}"
    test "${NATIVE_V2_TRAIN_PHASE:-}" = gradient-audit
    test "${NATIVE_V2_FULL24_PER8:-0}" = 0
    test -z "${NATIVE_V2_RESUME_JOB:-}${NATIVE_V2_EXTENSION_REVIEW:-}${NATIVE_V2_LEARNING_RATE:-}"
    SCRIPT="$DL_TASK_SOURCE/evals/native-v2-train.slurm" ;;
  native-teacher-smoke)
    : "${DL_TASK_SOURCE:?Immutable source required}"
    : "${DL_TASK_COMMIT:?Immutable commit required}"
    : "${CONTEXT_ACTUAL_NATIVE_TEACHER_APPROVAL_JOB:?Independent native input approval required}"
    : "${CONTEXT_ACTUAL_NATIVE_TEACHER_PREFLIGHT_JOB:?Independent numerical preflight required}"
    test "${CONTEXT_ACTUAL_TEMPORAL_FIT:-0}" = 1
    test "${CONTEXT_ACTUAL_FIT_UPDATES:-0}" = 8
    test "${CONTEXT_ACTUAL_RETENTION_TEACHER_WEIGHT:-0}" = 1
    test -z "${CONTEXT_RESUME_TRAINING_JOB:-}"
    SCRIPT="$DL_TASK_SOURCE/evals/context-training-proof.slurm" ;;
  teacher-engine-input)
    : "${DL_TASK_SOURCE:?Immutable source required}"
    : "${DL_TASK_COMMIT:?Immutable commit required}"
    : "${CONTEXT_TEACHER_SCOPE_JOB:?Independent teacher scope required}"
    : "${CONTEXT_TEACHER_ENGINE_INPUT_INDEX:?Single teacher index required}"
    [[ "$CONTEXT_TEACHER_ENGINE_INPUT_INDEX" =~ ^[0-9]+$ ]]
    SCRIPT="$DL_TASK_SOURCE/evals/context-training-proof.slurm" ;;
  rag) SCRIPT=evals/moshi-rag-training-proof.slurm ;;
  rag-reload) SCRIPT=evals/moshi-rag-reload-only.slurm ;;
  recovery-data) SCRIPT=evals/recovery-prepare-data.slurm ;;
  recovery-encode) SCRIPT=evals/recovery-encode.slurm ;;
  recovery-pilot)
    test "${RECOVERY_STEPS:-32}" -le 32
    SCRIPT=evals/recovery-train.slurm ;;
  recovery-voice) SCRIPT=evals/recovery-voice.slurm ;;
  recovery-audit) SCRIPT=evals/recovery-speech-audit.slurm ;;
  reference-effect) SCRIPT=evals/moshi-reference-effect.slurm ;;
  personaplex) SCRIPT=evals/personaplex-recovery.slurm ;;
  personaplex-paced) SCRIPT=evals/personaplex-paced.slurm ;;
  personaplex-reference) SCRIPT=evals/personaplex-reference.slurm ;;
  personaplex-bridge) SCRIPT=evals/personaplex-bridge-pilot.slurm ;;
  personaplex-worker) SCRIPT=evals/personaplex-worker.slurm ;;
  personaplex-history) SCRIPT=evals/personaplex-history.slurm ;;
  voice-quality) SCRIPT=evals/voice-quality-probe.slurm ;;
  codec-decode) SCRIPT=evals/voice-codec-decode.slurm ;;
  voice-audio-debug) SCRIPT=evals/voice-audio-debug.slurm ;;
  minicpm) SCRIPT=evals/minicpm-recovery-probe.slurm ;;
  lychee) SCRIPT=evals/lychee-voice-probe.slurm ;;
  qwen) SCRIPT=evals/qwen-delegation-probe.slurm ;;
  *) exit 2 ;;
esac
scontrol show job "$SLURM_JOB_ID"
# Direct bash execution reuses the committed payload. Its SBATCH comments
# are not interpreted here, so its workq directive cannot override reservation.
bash "$SCRIPT"
