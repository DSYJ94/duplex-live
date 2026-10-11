#!/bin/bash
#SBATCH --job-name=dl-msh
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/dl-sc1train-%j.out

# MERGED-CORPUS SELF-HISTORY ROUND (2026-10-11, full-speed restructure:
# gates pulled 5 days earlier; parallel dose arms on spare compute;
# 434-lesson corpus; held-out = q2 113-120 for direct comparability).
# Arm via SH_DOSE (0.5 default, 0.8 parallel). SUCCESS = Gate 1. D1-rerun
# verdict (job 7241355): readout path CLOSED even on fixed ears
# (LOO AUC 0.53-0.61 — chance). Decode fixes ruled out (hygiene
# arms). The ONE remaining lever for free-run words: train on the
# model's OWN outputs — --self-history 0.5, recollect mode (probe-11
# design), everything else identical to q2. Success metric: the
# end-of-diagnostic FREE-RUN WORD WER (q2 baseline ~0.96).
# Recipe held IDENTICAL to probe 9 (placement-perfect, the one proven
# point): v2 labels + word-weight 3.0 + self-history 0 (pure teacher
# forcing). Only QUANTITY changes: ~224 filter-passed lessons instead
# of 4, so each lesson is seen ~9x instead of ~100x — the five-point
# law's remedy (state coverage) without re-learning placement.
# Held-out: the LAST 8 per class (idx 113-120) never enter training;
# the closing free-run diagnoses on 4 of them.
# Resume: --save-every 200 keeps trained rows on disk; if the job dies,
# resubmit with TRAIN_INIT=$DL/runs/scale1-pt1/p1_checkpoint.pt to
# continue from the last checkpoint.
# Calibrate before first submit: compare probe 9's actual elapsed
# (sacct -j 6905339) to size STEPS vs this 24h budget.
# SUBMITTED BY evals/scale1-shard.slurm — do not launch by hand unless
# the chain is broken. FULL LOG — no grep at run time.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs-new:$REPO

SHARDS=$DL/runs/shards-merged
BATCH=$DL/runs/corpus-merged
OUT=$DL/runs/merged-sh${SH_DOSE}
STEPS=${STEPS:-4000}
INIT=${TRAIN_INIT:-}

cd "$REPO"

# train whitelist = per class idx 001-112 (held-out: 113-120)
TRAIN_LIST=""
for i in $(seq -w 1 112); do
    TRAIN_LIST="$TRAIN_LIST,q2_dense_dialog-$i,q2_dense_barge-$i"
done
for p in a1 a2 a3; do
    for d in $DL/runs/shards-merged/${p}_*.pt; do
        TRAIN_LIST="$TRAIN_LIST,$(basename $d .pt)"
    done
done
TRAIN_LIST=${TRAIN_LIST#,}

INIT_ARGS=""
[ -n "$INIT" ] && INIT_ARGS="--init-ckpt $INIT"

echo "===== scale1 train: $STEPS steps, ww3, SH0, $(echo "$TRAIN_LIST" | tr ',' '\n' | wc -l | tr -d ' ') lessons ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/duplex-live/pytorch-new.sif \
    python -m layera.train_p1 --shards "$SHARDS" --batch "$BATCH" \
    --steps "$STEPS" --word-weight 3.0 --save-every 200 \
        --self-history ${SH_DOSE} --sh-mode recollect \
    --lessons "$TRAIN_LIST" $INIT_ARGS \
    --out "$OUT"

echo "===== free-run on held-out lessons (diagnostic: pt1 alone is expected to murmur; gates come after) ====="
# pick up to 2 held-out lessons per class (idx 113-120) that actually
# HAVE shards — the filter can kill specific lessons, and a fixed name
# list crashed the diagnostic in round 1 (only 8 shards existed)
HELD=""
for cls in dense_dialog dense_barge; do
    n=0
    for i in $(seq 113 120); do
        id=$(printf '%s-%03d' "$cls" "$i")
        if [ -f "$SHARDS/$id.pt" ] && [ "$n" -lt 2 ]; then
            HELD="$HELD,$id"; n=$((n + 1))
        fi
    done
done
HELD=${HELD#,}
[ -n "$HELD" ] || { echo "no held-out shards found in $SHARDS"; exit 1; }
echo "held-out freerun lessons: $HELD"
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/duplex-live/pytorch-new.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt "$OUT/p1_checkpoint.pt" --shards "$SHARDS" \
    --scenarios "$HELD" --max-ticks 900
echo "scale1 train done"
