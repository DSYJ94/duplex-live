#!/bin/bash
#SBATCH --job-name=dl-sc1g1
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-sc1g1-%j.out

# GATE ROUND 1 ON THE SCALED TRUNK (2026-10-01, per the scale1 readout
# plan: "gate DAgger on the scaled model, train_gate --init-ckpt
# runs/scale1-pt1"). Scale1 freerun verdict (job 6971323): FAIL both
# bar halves ungated — 5/17 exact onsets, false speech 41-75% (3
# chatter lessons) + 1 mute lesson. The five-point law's question for
# the scaled states: can a DAgger gate on the 33x-coverage trunk split
# turn-start from chatter where the dense-pt3 trunk could not?
# Round 1 = UNGATED rollout (no --dagger-gate): collect word-first
# proposal states exactly as the raw base model visits them (the 91%
# quiet-word prior lives here), label gold, train gate 200 steps.
# 8 rollout lessons (4 dialog + 4 barge, ~12 min/lesson); sweep mirrors
# gate rounds 1-3 (3 thresholds on trained, 0.5 on held) for direct
# comparison. Pass bar v2: >=90% exact onsets AND false speech <10%
# of quiet ticks. FULL LOG — no grep at run time.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-scale1-q
BATCH=$DL/runs/synth-scale1-q/audio
BASE=$DL/runs/scale1-q-pt1/p1_checkpoint.pt
GATE_OUT=$DL/runs/scale1-q-gate1

# rollout+train lessons: 4 dialog + 4 barge from the scale1 train split
ROLL="dense_dialog-001,dense_dialog-010,dense_dialog-012,dense_dialog-020,dense_barge-001,dense_barge-004,dense_barge-007,dense_barge-010"
# sweep lessons: 2+2 of the above; held = the SAME 4 the scale1 train
# freerun diagnosed (direct comparison with the ungated numbers)
TRAIN4="dense_dialog-001,dense_dialog-010,dense_barge-001,dense_barge-004"
HELD4="dense_dialog-116,dense_dialog-118,dense_barge-113,dense_barge-114"

cd "$REPO"

echo "===== scale1 gate round 1: 8-lesson UNGATED dagger rollout ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.train_gate --shards "$SHARDS" --batch "$BATCH" \
    --steps 200 --corrupt-prob 0.5 --dagger \
    --buffer-batch 512 \
    --lessons "$ROLL" \
    --init-ckpt "$BASE" \
    --out "$GATE_OUT"

for TH in 0.35 0.5 0.65; do
  echo "===== free-run scale1+gate1 thr=$TH on trained lessons ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m layera.freerun_diag --batch "$BATCH" \
      --ckpt "$BASE" --shards "$SHARDS" \
      --scenarios "$TRAIN4" --max-ticks 900 \
      --gate-ckpt $GATE_OUT/gate_checkpoint.pt \
      --gate-threshold "$TH"
done

echo "===== free-run scale1+gate1 thr=0.5 on heldout lessons ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.freerun_diag --batch "$BATCH" \
    --ckpt "$BASE" --shards "$SHARDS" \
    --scenarios "$HELD4" --max-ticks 900 \
    --gate-ckpt $GATE_OUT/gate_checkpoint.pt \
    --gate-threshold 0.5
echo "scale1 gate1 done"
