#!/bin/bash
#SBATCH --job-name=dl-m2q2
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-m22b-%j.out

# M2 milestone 2b bench (2026-10-02): decisions on time, pieces in
# arrears. The scheduler lives in m2/parity.py (--piece-mode arrears):
# the ear-frame decision is the only latency-critical work per tick;
# the model's word tokens drain on later ticks under a fixed budget
# (--pieces-per-tick), before the next ear frame so cache order tracks
# the inline order whenever the backlog fits. This job measures the
# grid {piece-mode x pieces-per-tick x MoE kernel} on TRAIN4 shards:
# per arm we get adherence (the 80 ms budget), decision-tick latency,
# backlog, dry-vs-paced parity, and argmax divergence vs the inline
# reference picks. ENGINEERING BENCH ONLY — it measures the loop's
# clock behaviour, it is NOT a model-quality verdict (the fail-closed
# data rule still governs when those may be drawn).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

CKPT=$DL/runs/scale1-q2-pt1/p1_checkpoint.pt
SHARDS=$DL/runs/shards-scale1-q2
TRAIN4="dense_dialog-001,dense_dialog-010,dense_barge-001,dense_barge-004"
OUT=$DL/runs/m2-q2-smoke
REF=$DL/runs/m2-2b-bench/picks-inline

cd "$REPO"
mkdir -p "$OUT"

run_arm () {
  TAG="$1"; shift
  echo "===== 2b arm: $TAG ====="
  # an arm whose gates are not all green still delivers its numbers —
  # the point of the bench is the measurement, expected-fail on
  # adherence is a RESULT here, not a crashed job
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m m2.parity --ckpt "$CKPT" --shards "$SHARDS" \
      --scenarios "$TRAIN4" --tick-ms 80 --out "$OUT/$TAG" \
      "$@" \
      || echo "[bench] arm $TAG: gates not all green (numbers above are the result)"
}

run_arm inline-ref --piece-mode inline --experts batched_mm \
    --save-picks "$OUT/picks-inline"
run_arm arrears-ppt4-mm --piece-mode arrears --pieces-per-tick 4 \
    --experts batched_mm --ref-picks "$REF/picks.json"
run_arm arrears-ppt4-mm --piece-mode arrears --pieces-per-tick 4 \
    --experts batched_mm --ref-picks "$REF/picks.json"
echo "m2 q2 smoke done"
