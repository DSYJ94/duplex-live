#!/bin/bash
#SBATCH --job-name=synth-render
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/synth-render-%j.out
#
# Synthetic factory v0 (Issue #3): generate timelines (CPU) then render them
# with MiniCPM-o 4.5 TTS under the pylibs-minicpm overlay (transformers 4.51).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache
export TMPDIR=$DL/tmp
ENGINE=${ENGINE:-minicpm}   # minicpm | chatterbox
case "$ENGINE" in
  minicpm)    OVERLAY=$DL/pylibs-minicpm:$DL/pylibs ;;
  chatterbox) OVERLAY=$DL/pylibs-chatterbox:$DL/pylibs-minicpm:$DL/pylibs ;;
  *) echo "unknown ENGINE $ENGINE"; exit 1 ;;
esac
export PYTHONPATH=$OVERLAY:$REPO
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

PER_CLASS=${PER_CLASS:-4}
SEED=${SEED:-17}
OUT_BASE=${OUT_BASE:-runs/synth}   # e.g. OUT_BASE=runs/synth-v02 sbatch ...
TL_DIR=$DL/$OUT_BASE/timelines
OUT_DIR=$DL/$OUT_BASE/audio

APPT="apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER"

cd "$REPO"

echo "[synth] generating timelines -> $TL_DIR"
mkdir -p "$TL_DIR" "$OUT_DIR"
SC_ARGS=""
[ -n "${BANKS:-}" ] && [ -f "$BANKS" ] && SC_ARGS="--banks $BANKS"
$APPT python -m data.synthetic.scenarios --out "$TL_DIR" --per-class "$PER_CLASS" --seed "$SEED" $SC_ARGS

ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
CACHE=${CACHE:-$DL/runs/clip-cache}
VARGS=""
[ -f "$VJ" ] && VARGS="--voices $VJ --anchor-prob ${ANCHOR_PROB:-0.5}"
echo "[synth] rendering with engine=$ENGINE, ref voices from $ASSETS -> $OUT_DIR (pool: $VARGS, cache: $CACHE)"
$APPT python -m data.synthetic.render \
    --timelines "$TL_DIR" --out "$OUT_DIR" --assets "$ASSETS" --engine "$ENGINE" \
    --clip-cache "$CACHE" $VARGS

echo "[synth] done. manifest: $OUT_DIR/manifest.jsonl"

# self-chain the quality filter (afterok) so batches finish end-to-end
# without an outside driver — cert lapses no longer stall the pipeline
FILTER_JOB=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,AUDIO=$OUT_DIR \
    -o $DL/logs/synth-filter-%j.out repo/synth-filter.slurm)
echo "[synth] chained filter job: $FILTER_JOB (afterok:$SLURM_JOB_ID)"
