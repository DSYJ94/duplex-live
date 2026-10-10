#!/bin/bash
#SBATCH --job-name=synth-filter
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/synth-filter-%j.out
#
# Synthetic factory quality filter (Issue #3): rule checks + open-ASR
# round-trip (whisper-large-v3-turbo, MIT) over a rendered batch.
# Reads runs/synth/audio/{manifest.jsonl,*}; writes filter_report.jsonl +
# train_manifest.jsonl next to the manifest. Runs under the main pylibs
# overlay (transformers 5.17) — no MiniCPM deps needed.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache
export TMPDIR=$DL/tmp
export PYTHONPATH=$DL/pylibs-new:$REPO
CONTAINER=/projects/u6si/duplex-live/pytorch-new.sif

AUDIO=${AUDIO:-$DL/runs/synth/audio}
ASR_MODEL=${ASR_MODEL:-openai/whisper-large-v3-turbo}

APPT="apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER"

cd "$REPO"

echo "[filter] judging $AUDIO with $ASR_MODEL"
$APPT python -m data.synthetic.filter --audio "$AUDIO" --asr-model "$ASR_MODEL"

# close the clip-cache loop: gate-failed short clips get purged so they
# re-roll on the next render; passing clips stay frozen (no-op if unused)
CACHE=${CACHE:-$DL/runs/clip-cache}
if [ -d "$CACHE" ]; then
    $APPT python -m data.synthetic.purge_cache --cache "$CACHE" \
        --reports "$AUDIO/filter_report.jsonl" || true
fi

echo "[filter] done. report: $AUDIO/filter_report.jsonl"
