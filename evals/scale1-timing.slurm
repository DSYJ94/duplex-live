#!/bin/bash
#SBATCH --job-name=dl-sc1qt
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-sc1qt-%j.out

# TIMING AUDIT + QUALITY STAMP (fail-closed data rule, part 2 — owner
# directive 2026-10-02: "Make sure we dont go ahead when there are failures
# in the ... timing labels"). Whisper word timestamps vs the answer key's
# grid placement, per conversation; then the stamp folds the per-clip clip
# verdicts and the timing verdicts into quality.jsonl — the file every
# training/eval job asserts on at startup. STOPS after stamping: re-sharding
# and retraining are deliberate next steps on the owner's go.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
export PYTHONPATH=$DL/pylibs:$REPO
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

AUDIO=${AUDIO:?AUDIO env var required (batch dir with train_manifest.jsonl)}
BAR=${BAR:-200}   # ms; p95 answer-key-vs-audio offset a conversation tolerates

APPT="apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER"

cd "$REPO"

echo "[timing] auditing $AUDIO (bar p95 <= ${BAR}ms; re-label from measured onsets ON)"
$APPT python -m data.synthetic.timing_audit --batch "$AUDIO" \
    --offset-bar-ms "$BAR" --relabel

echo "[timing] stamping quality records (per-clip bar + timing) -> quality.jsonl"
$APPT python -m data.synthetic.quality --batch "$AUDIO" --timing "$AUDIO/timing_report.jsonl"

echo "[timing] done. $AUDIO/quality.jsonl is the gate — consumers abort without it."
