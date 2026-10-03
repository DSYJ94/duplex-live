#!/bin/bash
#SBATCH --job-name=dl-m2cr
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/dl-m2cr-%j.out

# M2 VOICE WALL, attempt 2 (2026-10-03). Attempt 1 (job 7032343):
#   compile@1gpu  -> CUDA OOM (compile buffers on top of the 30B model
#                    filled 95GB; 883MB reserved-but-unallocated = heavy
#                    fragmentation)
#   shard4+compile-> 785ms p50 (WORSE: pipeline-sharding adds per-token
#                    interconnect hops)
#   shard4-plain  -> 489ms p50 (also worse than 1-GPU 386ms)
# Sharding is dead for latency; the lever is compile on ONE GPU. This
# retry adds PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (defrags
# the allocator; the OOM'd alloc was 48MB with 23MB free) and
# TORCHINDUCTOR_COMPILE_THREADS to bound compile workspace. Baseline to
# beat: 386ms p50 speaking tick, 59ms decision tick, 80ms/word marginal.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_COMPILE_THREADS=16
export TORCHINDUCTOR_CACHE_DIR=$DL/tmp/inductor-cache

CKPT=$DL/runs/scale1-pt1/p1_checkpoint.pt
SHARDS=$DL/runs/shards-scale1
OUT=$DL/runs/m2-voice5x
REF=$OUT/picks-inline

cd "$REPO"
mkdir -p "$OUT" "$DL/tmp/inductor-cache"

echo "===== compile retry: 1 GPU, expandable_segments ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env CUDA_VISIBLE_DEVICES=0 --env PYTHONPATH=$PYTHONPATH \
    --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    --env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
    --env TORCHINDUCTOR_COMPILE_THREADS=$TORCHINDUCTOR_COMPILE_THREADS \
    --env TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m m2.parity --ckpt "$CKPT" --shards "$SHARDS" \
    --scenarios dense_dialog-001 --tick-ms 80 --out "$OUT/compile1gpu-es" \
    --piece-mode arrears --pieces-per-tick 4 --experts batched_mm \
    --ref-picks "$REF/picks.json" --compile \
    || echo "[5x] compile retry: gates not all green (numbers above are the result)"
echo "compile retry done"
