#!/bin/bash
#SBATCH --job-name=dl-m25x
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-m25x-%j.out

# M2 VOICE WALL attack (owner approval 2026-10-03: "approved on the 5x
# compute needed, use isambard"). The 2b bench measured the wall: a
# speaking tick on dense_dialog-001 costs p50 386 ms (arrears ppt4,
# batched_mm, ONE GH200) against the 80 ms budget — ~80 ms per word
# token for a ~3B-active-param MoE, which is ~10x worse than serving-
# grade for this hardware: the gap is per-call overhead, not FLOPs.
# This job measures the two stacked levers the approval buys:
#   A. torch.compile (parity.py --compile, dynamic shapes) on 1 GPU
#   B. model sharded across 4 GH200s (device_map=auto) + compile
#   C. 4 GH200s, no compile (isolates the sharding effect)
# against the recorded baseline (job 7026771: 386 ms p50 speaking,
# 59 ms p50 decision, backlog max ~1300 on this lesson). dense_dialog-001
# ONLY — it is the wall. Engineering bench: parity must stay 0.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

CKPT=$DL/runs/scale1-pt1/p1_checkpoint.pt
SHARDS=$DL/runs/shards-scale1
OUT=$DL/runs/m2-voice5x
REF=/projects/u6si/duplex-live/runs/m2-2b-bench/picks-inline

cd "$REPO"
mkdir -p "$OUT"

run_arm () {
  TAG="$1"; NGPU="$2"; shift 2
  echo "===== 5x arm: $TAG (CUDA_VISIBLE_DEVICES=$NGPU) ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env CUDA_VISIBLE_DEVICES=$NGPU \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m m2.parity --ckpt "$CKPT" --shards "$SHARDS" \
      --scenarios dense_dialog-001 --tick-ms 80 --out "$OUT/$TAG" \
      --piece-mode arrears --pieces-per-tick 4 --experts batched_mm \
      --ref-picks "$REF/picks.json" \
      "$@" \
      || echo "[5x] arm $TAG: gates not all green (numbers above are the result)"
}

run_arm compile-1gpu 0 --compile
run_arm shard4-compile 0,1,2,3 --compile
run_arm shard4-plain 0,1,2,3
echo "m2 voice-5x bench done"
