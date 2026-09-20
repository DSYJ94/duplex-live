#!/bin/bash
#SBATCH --job-name=layera-m1
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/layera-m1-%j.out
#
# Layer-A M1 (Issue #7): tick-loop driver on unmodified Qwen3-Omni.
# Feeds a factory user-stem beat by beat; audio keeps entering DURING
# generation (the thing stock serving never does). pylibs overlay
# (transformers 5.17, qwen-omni-utils); batched_mm is mandatory on GH200.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

STEM=${STEM:-$DL/runs/synth-big2/audio/barge_in-000/stem_user.wav}
OUT=${OUT:-$DL/runs/layera-m1/report.json}
mkdir -p $DL/runs/layera-m1

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -u -m layera.driver --stem "$STEM" --endpoint-s ${ENDPOINT_S:-4.0} \
    --max-s ${MAX_S:-12.0} --out "$OUT"

echo "[m1] report: $OUT"
