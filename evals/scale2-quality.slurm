#!/bin/bash
#SBATCH --job-name=dl-sc2q
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/dl-sc2q-%j.out

# DATA TOP-UP for the GPT-live push (owner directive 2026-10-06).
# Dialog-heavy: 240 new dialogs (yield ~40% -> ~95 verified) + 60 new
# barges -> combined corpus ~115+55 barge / 48+95 dialog — roughly
# balanced, dialogs no longer the starved class. New seed (29) over
# the SAME content banks; the whole fail-closed chain applies (judge
# + trim + repair 3 + sampled re-speaks -> filter -> timing+relabel
# -> stamp). Self-chains, cert-lapse-proof.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

BANKS=$DL/runs/banks/banks-scale1.json
TL_DIR=$DL/runs/synth-scale2/timelines
OUT_DIR=$DL/runs/synth-scale2/audio
CACHE=$DL/runs/clip-cache

cd "$REPO"
mkdir -p "$TL_DIR" "$OUT_DIR" "$DL/logs"

echo "[scale2] generating dialogs (seed 29): 240"
export PYTHONPATH=$DL/pylibs:$REPO
apptainer exec --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER \
    python -m data.synthetic.scenarios --out "$TL_DIR" \
    --per-class 240 --seed 29 --banks "$BANKS" --classes dense_dialog
echo "[scale2] generating barges (seed 31): 60"
apptainer exec --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER \
    python -m data.synthetic.scenarios --out "$TL_DIR" \
    --per-class 60 --seed 31 --banks "$BANKS" --classes dense_barge

export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO
ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
VARGS=""
[ -f "$VJ" ] && VARGS="--voices $VJ --anchor-prob 0.5"

echo "[scale2] rendering 300 lessons with judge+trim+repair 3 (2 workers)"
pids=""
for k in 0 1; do
    apptainer exec --nv --bind /projects,/lus,/home \
        --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
        $CONTAINER \
        python -m data.synthetic.render \
        --timelines "$TL_DIR" --out "$OUT_DIR" --assets "$ASSETS" \
        --engine minicpm --clip-cache "$CACHE" --slice "$k/2" $VARGS \
        --repair 3 \
        > "$DL/logs/render-scale2-w$k.log" 2>&1 &
    pids="$pids $!"
done
fail=0
for p in $pids; do wait "$p" || fail=1; done
[ $fail = 1 ] && { echo "[scale2] a render worker failed — see logs/render-scale2-w*.log"; exit 1; }
cat $OUT_DIR/manifest.part*of2.jsonl >> $OUT_DIR/manifest.jsonl 2>/dev/null || true
rm -f $OUT_DIR/manifest.part*of2.jsonl
echo "[scale2] render done: $(wc -l < $OUT_DIR/manifest.jsonl) shipped"

FILTER_JOB=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,AUDIO=$OUT_DIR,CACHE=$CACHE \
    -o $DL/logs/synth-filter-sc2-%j.out $DL/repo/synth-filter.slurm)
echo "[scale2] chained filter job: $FILTER_JOB"
TS_JOB=$(sbatch --parsable --dependency=afterok:$FILTER_JOB \
    --export=ALL,AUDIO=$OUT_DIR \
    -o $DL/logs/dl-sc2qt-%j.out $DL/repo/evals/scale1-timing.slurm)
echo "[scale2] chained timing+stamp job: $TS_JOB"
echo "[scale2] data stage done — quality.jsonl lands in $OUT_DIR"
