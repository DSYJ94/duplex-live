#!/bin/bash
#SBATCH --job-name=dl-m2p2a
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/dl-m2p2a-%j.out

# M2.2a (owner decision 2026-09-28: "both in parallel"): tick parity on
# the SERVING decode config — pt3 + activity gate2 @0.5 + user-speech
# mask (hold 3) — the config M2 would actually run live. Three jobs in
# one: (1) baseline batched_mm = latency truth for the serving config
# (gate MLP cost included) + picks.json saved as THE cross-config
# reference the live loop must match (docs/04 shadow-test pattern);
# (2) --experts eager (docs/04 favours gather+GEMV for batch-1; the
# pt1-era bench said batched_mm — re-measure at pt3+gate);
# (3) --compile (dynamic; win measured, not assumed). The known wall
# this quantifies: piece-loop forwards during speech (~57 ms/forward
# at pt1) vs the 80 ms tick. Gates per lesson: adherence >= 99.9%,
# dry-vs-paced parity = 0, p999 < 80 ms. FULL LOG — no grep.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
CKPT=$DL/runs/dense-pt3/p1_checkpoint.pt
GATE=$DL/runs/dense-gate2/gate_checkpoint.pt
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"

cd "$REPO"

run_arm () {  # $1 tag; the rest = extra freerun flags, verbatim
  local TAG=$1; shift
  echo "===== m2.2a parity arm $TAG ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m m2.parity --ckpt "$CKPT" --shards "$SHARDS" \
      --scenarios "$TRAIN4" --gate-ckpt "$GATE" --gate-threshold 0.5 \
      --user-gate --gate-hold 3 --batch "$BATCH" \
      --out $DL/runs/m2-parity-serving-$TAG "$@"
}

run_arm batched --save-picks $DL/runs/picks-m2-serving
run_arm eager --experts eager --ref-picks $DL/runs/picks-m2-serving/picks.json
run_arm compile --compile --ref-picks $DL/runs/picks-m2-serving/picks.json

echo "m2.2a serving parity done"
