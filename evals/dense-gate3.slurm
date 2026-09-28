#!/bin/bash
#SBATCH --job-name=dl-gate3
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-gate3-%j.out

# GATE ROUND 3 (owner decision 2026-09-28: "both in parallel" with M2).
# Round 2 (DAgger) fixed onset ranking (16/17 exact at 0.5) but the
# quiet side did not generalize: buffer was 2400 states from only the
# 4 TRAIN4 lessons vs 529k gate params (memorization territory; held
# chatter 52-82%). Round 3's one big change: rollout + train on ALL 12
# training shards (~3x state diversity) instead of 4, rollout with
# gate2 ACTIVE at its deployed threshold 0.5 (iterated DAgger: the
# states THIS decode config visits), buffer batch 256 -> 512. Sweep
# identical to rounds 1/2 for direct comparison. Pass bar v2: >= 90%
# exact onsets AND false speech < 10% of quiet ticks. FULL LOG — no
# grep at run time.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
BATCH=$DL/runs/synth-dense1/audio
TRAIN4="dense_dialog-001,dense_dialog-004,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-002,dense_dialog-005,dense_barge-002,dense_barge-005"

cd "$REPO"

echo "===== gate round 3: 12-shard dagger rollout (gate2 active @0.5) ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_gate --shards "$SHARDS" --batch "$BATCH" \
    --steps 200 --corrupt-prob 0.5 --dagger \
    --dagger-gate $DL/runs/dense-gate2/gate_checkpoint.pt \
    --rollout-threshold 0.5 --buffer-batch 512 \
    --init-ckpt $DL/runs/dense-pt3/p1_checkpoint.pt \
    --out $DL/runs/dense-gate3

for TH in 0.35 0.5 0.65; do
  echo "===== free-run pt3+gate3 thr=$TH on trained lessons ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
      --scenarios "$TRAIN4" --max-ticks 900 \
      --gate-ckpt $DL/runs/dense-gate3/gate_checkpoint.pt \
      --gate-threshold $TH
done

echo "===== free-run pt3+gate3 thr=0.5 on heldout lessons ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt $DL/runs/dense-pt3/p1_checkpoint.pt --shards "$SHARDS" \
    --scenarios "$HELD4" --max-ticks 900 \
    --gate-ckpt $DL/runs/dense-gate3/gate_checkpoint.pt \
    --gate-threshold 0.5
echo "dense gate 3 done"
