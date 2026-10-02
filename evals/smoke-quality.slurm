#!/bin/bash
#SBATCH --job-name=dl-smokeq
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/dl-smokeq-%j.out

# SMOKE for the fail-closed data chain (2026-10-02): one conversation
# through the render-time judge (--repair 1) and the timing audit on two
# OLD-batch conversations. Catches runtime breakage (whisper under the
# minicpm overlay, judge import, word timestamps) before the 12h
# re-render chain is submitted. Throwaway outputs under $DL/tmp/smoke-q.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif
cd "$REPO"
mkdir -p "$DL/tmp/smoke-q/audio" "$DL/tmp/smoke-q/cache"

ASSETS=$(ls -d $HF_HOME/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets | head -1)
VJ=$DL/models/voices/voices.json
VARGS=""
[ -f "$VJ" ] && VARGS="--voices $VJ --anchor-prob 0.5"

echo "===== smoke 1/2: render with judge (dense_dialog-000, --repair 1) ====="
export PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER \
    python -m data.synthetic.render \
    --timelines $DL/runs/synth-scale1/timelines --out $DL/tmp/smoke-q/audio \
    --assets "$ASSETS" --engine minicpm --only dense_dialog-000 \
    --repair 1 --clip-cache $DL/tmp/smoke-q/cache $VARGS

echo "===== smoke 2/2: timing audit on OLD batch (limit 2) ====="
export PYTHONPATH=$DL/pylibs:$REPO
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER \
    python -m data.synthetic.timing_audit --batch $DL/runs/synth-scale1 --limit 2

echo "smoke done"
