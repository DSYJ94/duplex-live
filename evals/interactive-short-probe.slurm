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
