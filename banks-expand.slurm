#!/bin/bash
#SBATCH --job-name=banks-expand
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/banks-expand-%j.out
#
# Open-LLM content-bank expansion (Issue #3, hours-scale composition).
# Qwen3-Omni-30B-A3B thinker, text-only, under the pylibs overlay
# (transformers 5.17; MoE on GH200 requires experts_implementation=
# "batched_mm" — baked into banks_llm.load_model). Output banks.json is a
# sidecar consumed by scenarios.py --banks; seeds stay canonical in-repo.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache
export TMPDIR=$DL/tmp
export PYTHONPATH=$DL/pylibs:$REPO
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

OUT=${OUT:-$DL/runs/banks/banks.json}
SEED=${SEED:-7}

APPT="apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    $CONTAINER"

cd "$REPO"

echo "[banks] expanding content banks -> $OUT"
$APPT python -m data.synthetic.banks_llm --out "$OUT" --seed "$SEED"

echo "[banks] done."
