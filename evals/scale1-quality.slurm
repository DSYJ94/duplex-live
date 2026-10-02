#!/bin/bash
#SBATCH --job-name=dl-sc1q
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/dl-sc1q-%j.out

# SCALING RUN, fail-closed edition (owner directive 2026-10-02: "fix the
# training data... dont go ahead when there are failures"). Re-renders the
# EXISTING scale1 timelines (same banks, same seed, same voice assignment)
# with the render-time judge ON: every clip is speech-to-text-checked as it
# is spoken (whisper-large-v3-turbo, same judge as the batch filter);
# failures re-speak (--repair 2); a clip that still fails REJECTS its
# conversation — no mix, no manifest row. Validated clip cache carries over.
# Self-chains, cert-lapse-proof:
#   filter job (afterok here) -> timing+stamp job (afterok filter)
# and STOPS — no training chain. Re-sharding is a deliberate next step on
# the owner's go, once quality.jsonl says how many lessons verified.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

SRC=$DL/runs/synth-scale1          # timelines from the original scale1 run
TL_DIR=$SRC/timelines
OUT_BASE=$DL/runs/synth-scale1-q   # NEW batch dir — the old one stays
OUT_DIR=$OUT_BASE/audio
CACHE=$DL/runs/clip-cache          # validated clips carry over; bad entries
                                   # purge themselves at render time now

cd "$REPO"
mkdir -p "$OUT_DIR" "$DL/logs"

export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO
APPT="apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER"

ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
VARGS=""
[ -f "$VJ" ] && VARGS="--voices $VJ --anchor-prob 0.5"

echo "[quality] re-render with judge: $TL_DIR -> $OUT_DIR (--repair 2, 2 workers)"
pids=""
for k in 0 1; do
    $APPT python -m data.synthetic.render \
        --timelines "$TL_DIR" --out "$OUT_DIR" --assets "$ASSETS" --engine minicpm \
        --clip-cache "$CACHE" --slice "$k/2" $VARGS --repair 2 \
        > "$DL/logs/render-q-w$k.log" 2>&1 &
    pids="$pids $!"
done
fail=0
for p in $pids; do wait "$p" || fail=1; done
[ $fail = 1 ] && { echo "[quality] a render worker failed — see logs/render-q-w*.log"; exit 1; }
cat $OUT_DIR/manifest.part*of2.jsonl >> $OUT_DIR/manifest.jsonl 2>/dev/null || true
rm -f $OUT_DIR/manifest.part*of2.jsonl
echo "[quality] render done: $(wc -l < $OUT_DIR/manifest.jsonl) conversations shipped (rejected ones wrote no manifest row)"

# --- self-chain: filter -> timing audit + quality stamp ---
FILTER_JOB=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,AUDIO=$OUT_DIR,CACHE=$CACHE \
    -o $DL/logs/synth-filter-q-%j.out $DL/repo/synth-filter.slurm)
echo "[quality] chained filter job: $FILTER_JOB (afterok:$SLURM_JOB_ID)"
TS_JOB=$(sbatch --parsable --dependency=afterok:$FILTER_JOB \
    --export=ALL,AUDIO=$OUT_DIR \
    -o $DL/logs/dl-sc1qt-%j.out $DL/repo/evals/scale1-timing.slurm)
echo "[quality] chained timing+stamp job: $TS_JOB (afterok:$FILTER_JOB)"
echo "[quality] done here. quality.jsonl lands in $OUT_DIR — that file gates every training job now."
