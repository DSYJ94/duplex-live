#!/bin/bash
#SBATCH --job-name=dl-sc1dgr
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/dl-sc1dgr-%j.out

# SCALING RUN, dialog top-up roll (2026-09-30): round 2 recovered only
# 5/101 dense_dialog lessons — tail_babble is systematic (MiniCPM babbles
# at clip ends with base probability; dialog lessons have the most
# assistant clips, so nearly every lesson rolls >=1 bad clip; no culprit
# voice). This roll re-renders ONLY the failing dialog timelines (their
# purged clips get fresh rolls; the 38 one-bad-clip lessons have good
# odds), then re-filters the whole audio dir. NO train/shard chain:
# whether recovered lessons enter training is an owner decision after
# the current run's held-out results (resume via TRAIN_INIT exists).
# Manifest is cleared (render appends; filter rebuilds it fully).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

TL_ALL=$DL/runs/synth-scale1/timelines
TL_DIR=$DL/runs/synth-scale1/timelines-dialogroll
OUT_DIR=$DL/runs/synth-scale1/audio
CACHE=$DL/runs/clip-cache
REPORT=$OUT_DIR/filter_report.jsonl

cd "$REPO"
export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO
APPT="apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER"

# failing dialog lessons from the CURRENT filter report (absolute paths:
# job cwd is $REPO, data lives under $DL/runs)
FAILING=$($APPT python - <<PY
import json
rep = [json.loads(l) for l in open("$DL/runs/synth-scale1/audio/filter_report.jsonl")]
ids = [r["scenario_id"] for r in rep
       if r["scenario_class"] == "dense_dialog" and r["verdict"] == "fail"]
print("\n".join(ids))
PY
)
N=$(printf '%s\n' "$FAILING" | grep -c .)
echo "[dialogroll] $N failing dialog lessons to re-roll"
[ "$N" -gt 0 ] || { echo "[dialogroll] nothing to do"; exit 0; }

# subset timeline dir (symlinks; scenarios are selected by file)
rm -rf "$TL_DIR"; mkdir -p "$TL_DIR"
for id in $FAILING; do
    ln -s "$TL_ALL/$id".* "$TL_DIR/" 2>/dev/null || true
done
echo "[dialogroll] timeline subset: $(ls "$TL_DIR" | wc -l) files"

echo "[dialogroll] saving non-re-rolled manifest rows (render appends; the "
echo "  subset render would otherwise leave manifest with ONLY dialog lessons)"
cp "$OUT_DIR/manifest.jsonl" "$OUT_DIR/manifest.keep.jsonl"
rm -f "$OUT_DIR"/manifest.part*of1.jsonl

ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
VARGS=""
[ -f "$VJ" ] && VARGS="--voices $VJ --anchor-prob 0.5"
$APPT python -m data.synthetic.render \
    --timelines "$TL_DIR" --out "$OUT_DIR" --assets "$ASSETS" --engine minicpm \
    --clip-cache "$CACHE" --slice 0/1 $VARGS \
    > "$DL/logs/render-scale1-dgr.log" 2>&1
# atomic reassembly: kept rows (non-failing) + fresh rows for re-rolled
# lessons. mv is atomic — a concurrently-running reader never sees a
# partial manifest.
$APPT python - <<PY
import json, os, glob
fail = set("""$FAILING""".split())
keep = [json.loads(l) for l in open("$OUT_DIR/manifest.keep.jsonl")]
kept = [r for r in keep if r["scenario_id"] not in fail]
part = []
for p in glob.glob("$OUT_DIR/manifest.part*of1.jsonl"):
    part += [json.loads(l) for l in open(p)]
ids = [r["scenario_id"] for r in kept + part]
assert len(ids) == len(set(ids)) == 240, f"manifest reassembly: {len(ids)} rows"
with open("$OUT_DIR/manifest.jsonl.tmp", "w") as f:
    for r in kept + part:
        f.write(json.dumps(r) + "\n")
os.replace("$OUT_DIR/manifest.jsonl.tmp", "$OUT_DIR/manifest.jsonl")
print(f"[dialogroll] manifest reassembled: {len(kept)} kept + {len(part)} re-rolled")
PY
rm -f "$OUT_DIR"/manifest.part*of1.jsonl "$OUT_DIR/manifest.keep.jsonl"
echo "[dialogroll] render done — re-filtering full audio dir"

FILTER_JOB=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,AUDIO=$OUT_DIR \
    -o $DL/logs/synth-filter-%j.out $DL/repo/synth-filter.slurm)
echo "[dialogroll] chained filter job: $FILTER_JOB (afterok:$SLURM_JOB_ID)"
echo "[dialogroll] done. NO shard/train chain — owner decision pending"
