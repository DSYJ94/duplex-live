#!/bin/bash
#SBATCH --job-name=dl-sc2tl
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-sc2tl-%j.out

# SCALE-2 CONTENT (2026-10-06, the GPT-live push): fresh banks (seed 11)
# + 480 dialog timelines (seed 33) for the ElevenLabs volume arm. No API
# key needed for this stage. The API render arm (evals/scale2-api-render
# .slurm) consumes these timelines once ELEVENLABS_API_KEY is placed at
# $DL/keys/elevenlabs.env.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

BANKS=$DL/runs/banks/banks-scale2.json
TL_DIR=$DL/runs/synth-scale2-api/timelines

cd "$REPO"
mkdir -p "$TL_DIR" "$(dirname "$BANKS")"

if [ ! -f "$BANKS" ]; then
    echo "[scale2tl] expanding content banks (seed 11) -> $BANKS"
    apptainer exec --nv --bind /projects,/lus,/home \
        --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
        $CONTAINER \
        python -m data.synthetic.banks_llm --out "$BANKS" --seed 11
else
    echo "[scale2tl] banks exist: $BANKS"
fi

echo "[scale2tl] generating 480 dialog timelines (seed 33)"
apptainer exec --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER \
    python -m data.synthetic.scenarios --out "$TL_DIR" \
    --per-class 480 --seed 33 --banks "$BANKS" --classes dense_dialog
echo "[scale2tl] done: $(ls $TL_DIR | wc -l) timelines in $TL_DIR"
