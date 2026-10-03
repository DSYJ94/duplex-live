#!/bin/bash
#SBATCH --job-name=dl-smokeq
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-smokeq-%j.out

# SMOKE for the full fail-closed chain (2026-10-02, rev 3): render TWO
# lessons (dialog + barge) with the judge and silence-trim ON, then
# filter -> timing audit -> quality stamp -> consumer assert, all on the
# fresh output. Proves every link the 12h re-render chain will run.
# Throwaway outputs under $DL/tmp/smoke-q.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif
cd "$REPO"
rm -rf "$DL/tmp/smoke-q"
mkdir -p "$DL/tmp/smoke-q/audio" "$DL/tmp/smoke-q/cache"

ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
VARGS=""
[ -f "$VJ" ] && VARGS="--voices $VJ --anchor-prob 0.5"
# function, not a variable: PYTHONPATH is re-exported per stage and the
# container must see the value current at CALL time
appt() {
    apptainer exec --nv --bind /projects,/lus,/home \
        --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
        $CONTAINER "$@"
}

echo "===== smoke 1/4: render with judge + trim (one dialog + one barge lesson) ====="
export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO
# NB: --only is a filename SUBSTRING filter — "dense_" matches ALL 240
# dense timelines (the rev-3 timeout lesson); name the lessons exactly
appt python -m data.synthetic.render \
    --timelines $DL/runs/synth-scale1/timelines --out $DL/tmp/smoke-q/audio \
    --assets "$ASSETS" --engine minicpm --only dense_dialog-003 \
    --repair 1 --clip-cache $DL/tmp/smoke-q/cache $VARGS
appt python -m data.synthetic.render \
    --timelines $DL/runs/synth-scale1/timelines --out $DL/tmp/smoke-q/audio \
    --assets "$ASSETS" --engine minicpm --only dense_barge-004 \
    --repair 1 --clip-cache $DL/tmp/smoke-q/cache $VARGS

echo "===== smoke 2/4: filter (independent re-verify) ====="
export PYTHONPATH=$DL/pylibs:$REPO
appt python -m data.synthetic.filter --audio $DL/tmp/smoke-q/audio

echo "===== smoke 3/4: timing audit + RELABEL (fresh, trimmed audio) ====="
appt python -m data.synthetic.timing_audit --batch $DL/tmp/smoke-q/audio --relabel

echo "===== smoke 4/4: quality stamp + consumer gate ====="
appt python -m data.synthetic.quality --batch $DL/tmp/smoke-q/audio \
    --timing $DL/tmp/smoke-q/audio/timing_report.jsonl
appt python - <<'PYEOF'
from data.synthetic.quality import assert_verified, DataNotVerified
try:
    recs = assert_verified("/projects/u6si/duplex-live/tmp/smoke-q/audio",
                           role="smoke consumer", require_timing=True)
    print(f"[smoke] GATE OPEN: {len(recs)} verified lessons — a consumer "
          f"would proceed")
except DataNotVerified as e:
    print(f"[smoke] GATE HELD (expected if timing bar is strict): {e}")
PYEOF

echo "smoke done"
