#!/bin/bash
#SBATCH --job-name=dl-sc1dat
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/dl-sc1dat-%j.out

# SCALING RUN stage 1 of 3 (owner decision 2026-09-29: "scaling run +
# M2"). The five-point law's remedy is state coverage: ~20x more dense
# practice conversations, then the proven probe-9 recipe on top. This
# job = the data factory: LLM content banks (Qwen3-Omni text-only,
# open weights — ADR-0002/0003: no closed-API content) -> timelines
# (dense_dialog + dense_barge ONLY: the two classes every probe/gate
# measured) -> MiniCPM-o TTS render (2 workers share the 96GB GPU).
# Then it self-chains, cert-lapse-proof, exactly like synth-render:
#   filter job (afterok here) -> shard job (afterok filter)
#   -> train job (chained by the shard job).
# Resume: a timeout only costs time — the clip cache resumes rendering
# and prepare skips already-sharded lessons. FULL LOG — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

PER_CLASS=${PER_CLASS:-120}
SEED=${SEED:-23}                       # 17 = the original synth-dense1
CLASSES=${CLASSES:-dense_dialog,dense_barge}
OUT_BASE=${OUT_BASE:-runs/synth-scale1}
BANKS=${BANKS:-$DL/runs/banks/banks-scale1.json}
CACHE=${CACHE:-$DL/runs/clip-cache}
WORKERS=${WORKERS:-2}                  # minicpm ~26GB each on 96GB HBM

TL_DIR=$DL/$OUT_BASE/timelines
OUT_DIR=$DL/$OUT_BASE/audio

cd "$REPO"

# --- content banks (plain pylibs overlay: banks_llm wants tf 5.17) ---
export PYTHONPATH=$DL/pylibs:$REPO
if [ ! -f "$BANKS" ]; then
    echo "[scale1] expanding content banks -> $BANKS"
    mkdir -p "$(dirname "$BANKS")"
    apptainer exec --nv --bind /projects,/lus,/home \
        --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
        $CONTAINER \
        python -m data.synthetic.banks_llm --out "$BANKS" --seed 7
else
    echo "[scale1] banks exist, reusing: $BANKS"
fi

# --- timelines + render (minicpm overlay) ---
export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO
APPT="apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER"

echo "[scale1] generating timelines: per-class=$PER_CLASS seed=$SEED classes=$CLASSES"
mkdir -p "$TL_DIR" "$OUT_DIR"
$APPT python -m data.synthetic.scenarios --out "$TL_DIR" \
    --per-class "$PER_CLASS" --seed "$SEED" --banks "$BANKS" --classes "$CLASSES"

ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
VARGS=""
[ -f "$VJ" ] && VARGS="--voices $VJ --anchor-prob 0.5"
echo "[scale1] rendering $((PER_CLASS * 2)) lessons -> $OUT_DIR ($WORKERS workers, cache: $CACHE)"

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
[ $fail = 1 ] && { echo "[scale1] a worker failed — see logs/render-scale1-w*.log"; exit 1; }
cat $OUT_DIR/manifest.part*of$WORKERS.jsonl >> $OUT_DIR/manifest.jsonl 2>/dev/null || true
rm -f $OUT_DIR/manifest.part*of$WORKERS.jsonl

# --- self-chain: filter -> shards -> training (cert-lapse-proof) ---
FILTER_JOB=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,AUDIO=$OUT_DIR \
    -o $DL/logs/synth-filter-%j.out $DL/repo/synth-filter.slurm)
echo "[scale1] chained filter job: $FILTER_JOB (afterok:$SLURM_JOB_ID)"
SHARD_JOB=$(sbatch --parsable --dependency=afterok:$FILTER_JOB \
    -o $DL/logs/dl-sc1shard-%j.out $DL/repo/evals/scale1-shard.slurm)
echo "[scale1] chained shard job: $SHARD_JOB (afterok:$FILTER_JOB)"
echo "[scale1] data stage done. manifest: $OUT_DIR/manifest.jsonl"
