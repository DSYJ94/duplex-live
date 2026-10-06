#!/bin/bash
#SBATCH --job-name=dl-sc2api
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/dl-sc2api-%j.out

# ELEVENLABS VOLUME ARM (2026-10-06, ADR-0003 Amended A: closed-API
# content permitted, provenance = engine tag in the manifest). Renders
# the scale2-api dialog timelines through the API engine with the
# fail-closed judge (whisper on this GPU; TTS itself is API — the GPU
# stays mostly idle, training jobs can coexist). Self-chains filter ->
# timing+relabel -> stamp. Volume via DIALOGS (default 120 ≈ $40 in
# credits; validate yield before scaling to 480+).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp

KEYFILE=$DL/keys/elevenlabs.env
[ -f "$KEYFILE" ] || { echo "[sc2api] NO KEY: create $KEYFILE with 'export ELEVENLABS_API_KEY=...' (chmod 600)"; exit 1; }
# shellcheck disable=SC1090
source "$KEYFILE"
[ -n "${ELEVENLABS_API_KEY:-}" ] || { echo "[sc2api] key file has no ELEVENLABS_API_KEY"; exit 1; }

export PYTHONPATH=$DL/pylibs:$REPO
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif
TL_DIR=$DL/runs/synth-scale2-api/timelines
OUT_DIR=$DL/runs/synth-scale2-api/audio
CACHE=$DL/runs/clip-cache
DIALOGS=${DIALOGS:-120}

cd "$REPO"
mkdir -p "$OUT_DIR" "$DL/logs"

FILES=$(ls "$TL_DIR"/*.json 2>/dev/null | head -n "$DIALOGS" | wc -l)
[ "$FILES" -gt 0 ] || { echo "[sc2api] no timelines in $TL_DIR (run scale2-timelines first)"; exit 1; }
echo "[sc2api] rendering $FILES dialogs via ElevenLabs (judge ON, repair 3)"

ls "$TL_DIR"/*.json | head -n "$DIALOGS" | while read -r f; do
    cp "$f" "$DL/tmp/sc2api-in-$(basename "$f")"
done
mkdir -p "$DL/tmp/sc2api-in"
ls "$TL_DIR"/*.json | head -n "$DIALOGS" | while read -r f; do
    cp "$f" "$DL/tmp/sc2api-in/"
done

for k in 0 1 2 3; do
    (
    apptainer exec --nv --bind /projects,/lus,/home \
        --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
        --env ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY \
        $CONTAINER \
        python -m data.synthetic.render \
        --timelines "$DL/tmp/sc2api-in" --out "$OUT_DIR" \
        --engine elevenlabs --clip-cache "$CACHE" --slice "$k/4" \
        --repair 3 \
        > "$DL/logs/render-sc2api-w$k.log" 2>&1
    ) &
done
wait
cat $OUT_DIR/manifest.part*of4.jsonl >> $OUT_DIR/manifest.jsonl 2>/dev/null || true
rm -f $OUT_DIR/manifest.part*of4.jsonl
echo "[sc2api] render done: $(wc -l < $OUT_DIR/manifest.jsonl) shipped"

FILTER_JOB=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID \
    --export=ALL,AUDIO=$OUT_DIR,CACHE=$CACHE \
    -o $DL/logs/synth-filter-sc2api-%j.out $DL/repo/synth-filter.slurm)
echo "[sc2api] chained filter job: $FILTER_JOB"
TS_JOB=$(sbatch --parsable --dependency=afterok:$FILTER_JOB \
    --export=ALL,AUDIO=$OUT_DIR \
    -o $DL/logs/dl-sc2apiqt-%j.out $DL/repo/evals/scale1-timing.slurm)
echo "[sc2api] chained timing+stamp job: $TS_JOB"
echo "[sc2api] done here — quality.jsonl will land in $OUT_DIR"
