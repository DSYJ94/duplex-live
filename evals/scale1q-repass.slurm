#!/bin/bash
#SBATCH --job-name=dl-sqrepass
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/dl-sqrepass-%j.out

# RECOVERY PASS over the render-REJECTED conversations (2026-10-03).
# 56 of 240 failed repair-in-render (a clip babbled through 2 re-speaks).
# Their VALIDATED clips are in the shared clip cache — only the failing
# clips re-roll, so this is cheap. --repair 3 gives one more dice roll.
# Renders into the SAME batch dir (appends manifest rows for the
# recovered; rewrites REJECTED.json for the still-unlucky), then
# self-chains filter -> timing+relabel+stamp so quality.jsonl reflects
# the recovered set. Runs alongside training (training reads SHARDS,
# not this batch — no interference; the shards are already built).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp

SRC=$DL/runs/synth-scale1
TL_DIR=$SRC/timelines
OUT_DIR=$DL/runs/synth-scale1-q/audio
CACHE=$DL/runs/clip-cache

cd "$REPO"
mkdir -p "$DL/tmp/repass-timelines"

# 1. collect rejected scenario ids -> symlink their timelines
IDS=$(ls -d "$OUT_DIR"/*/REJECTED.json 2>/dev/null | xargs -r -n1 dirname | xargs -r -n1 basename)
[ -n "$IDS" ] || { echo "[repass] nothing rejected — nothing to do"; exit 0; }
echo "[repass] recovering: $(echo $IDS | wc -w) conversations"
rm -f "$DL/tmp/repass-timelines"/*.json
for id in $IDS; do ln -sf "$TL_DIR/$id.json" "$DL/tmp/repass-timelines/$id.json"; done

# 2. render with judge + trim + one extra retry
export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO
ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
VARGS=""
[ -f "$VJ" ] && VARGS="--voices $VJ --anchor-prob 0.5"
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m data.synthetic.render \
    --timelines "$DL/tmp/repass-timelines" --out "$OUT_DIR" \
    --assets "$ASSETS" --engine minicpm --repair 3 --clip-cache "$CACHE" $VARGS

# 3. self-chain: filter -> timing+relabel+stamp (re-stamps the whole batch)
FILTER_JOB=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,AUDIO=$OUT_DIR,CACHE=$CACHE \
    -o $DL/logs/synth-filter-q2-%j.out $DL/repo/synth-filter.slurm)
echo "[repass] chained filter job: $FILTER_JOB"
TS_JOB=$(sbatch --parsable --dependency=afterok:$FILTER_JOB \
    --export=ALL,AUDIO=$OUT_DIR \
    -o $DL/logs/dl-sc1qt2-%j.out $DL/repo/evals/scale1-timing.slurm)
echo "[repass] chained timing+stamp job: $TS_JOB"
echo "[repass] done here"
