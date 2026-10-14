#!/bin/bash
# Finite payload inside the short interactive allocation.
set -euo pipefail
export CODEC_DEVICE=cuda
bash evals/voice-codec-decode.slurm
export VOICE_NAMES=synthetic-marin
export VOICE_CODEC_DIR=/projects/u6si/duplex-live/runs/voice-quality-7414595/codec
bash evals/voice-quality-probe.slurm
