#!/bin/bash
#SBATCH --job-name=dl-sc1rdr
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/dl-sc1rdr-%j.out

# SCALING RUN recovery (owner Option A, 2026-09-30): re-render the
# EXISTING scale1 timelines so the ~220 clips the filter purged from
# the clip cache get a fresh roll, then re-filter. Round 1 passed
# 137/240 but only 19/120 dense_dialog (tail_babble on individual
# clips, lesson WER fine) — purged keys re-roll, passing keys come
# back identical from cache, so this costs ~an hour, not a re-render.
# Skips banks + timeline generation (timelines on disk, seed 23).
# Chains filter -> shard (now --per-class 999, commit 9cc4213) ->
# train, cert-lapse-proof like the original chain.
# Manifest is CLEARED first: render appends manifest.part* to
# manifest.jsonl, so a rerun without clearing would duplicate rows.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

TL_DIR=$DL/runs/synth-scale1/timelines
OUT_DIR=$DL/runs/synth-scale1/audio
CACHE=$DL/runs/clip-cache
WORKERS=${WORKERS:-2}

cd "$REPO"
export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO
APPT="apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER"

[ -d "$TL_DIR" ] || { echo "[rerender] no timelines at $TL_DIR"; exit 1; }
echo "[rerender] clearing manifests (render appends; avoid duplicates)"
rm -f "$OUT_DIR"/manifest.jsonl "$OUT_DIR"/manifest.part*of$WORKERS.jsonl

ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
VARGS=""
[ -f "$VJ" ] && VARGS="--voices $VJ --anchor-prob 0.5"
echo "[rerender] rendering $(ls "$TL_DIR" | wc -l) timelines -> $OUT_DIR ($WORKERS workers, cache: $CACHE)"

pids=""
for k in $(seq 0 $((WORKERS - 1))); do
    $APPT python -m data.synthetic.render \
        --timelines "$TL_DIR" --out "$OUT_DIR" --assets "$ASSETS" --engine minicpm \
        --clip-cache "$CACHE" --slice "$k/$WORKERS" $VARGS \
        > "$DL/logs/render-scale1-w$k.log" 2>&1 &
    pids="$pids $!"
done
fail=0
for p in $pids; do wait "$p" || fail=1; done
[ $fail = 1 ] && { echo "[rerender] a worker failed — see logs/render-scale1-w*.log"; exit 1; }
cat $OUT_DIR/manifest.part*of$WORKERS.jsonl >> $OUT_DIR/manifest.jsonl 2>/dev/null || true
rm -f $OUT_DIR/manifest.part*of$WORKERS.jsonl

# --- self-chain: filter -> shards -> training (cert-lapse-proof) ---
FILTER_JOB=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,AUDIO=$OUT_DIR \
    -o $DL/logs/synth-filter-%j.out $DL/repo/synth-filter.slurm)
echo "[rerender] chained filter job: $FILTER_JOB (afterok:$SLURM_JOB_ID)"
SHARD_JOB=$(sbatch --parsable --dependency=afterok:$FILTER_JOB \
    -o $DL/logs/dl-sc1shard-%j.out $DL/repo/evals/scale1-shard.slurm)
echo "[rerender] chained shard job: $SHARD_JOB (afterok:$FILTER_JOB)"
echo "[rerender] done. manifest: $OUT_DIR/manifest.jsonl"
