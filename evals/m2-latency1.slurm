#!/bin/bash
#SBATCH --job-name=dl-m2lat1
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-m2lat1-%j.out

# M2 milestone 1.5 — tick-latency levers, measured not assumed
# (docs/04: "gather+GEMV for MoE instead of grouped GEMM" at small
# batch). Three parity runs on ONE lesson (dense_barge-004: short and
# the only TRAINED lesson that emits word pieces, so the piece-loop
# overrun shows): batched_mm (the probe/free-run kernel, reference),
# eager experts, eager + torch.compile(dynamic). Parity gate stays
# dry-vs-paced within each run; cross-kernel argmax divergence vs the
# reference is REPORTED (numerics differ slightly between kernels —
# the serving path should be chosen with that cost on the table).
# A variant that dies must not kill the rest — each is marked, the
# log carries all. FULL LOG — no grep pipes.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

SHARDS=$DL/runs/shards-dense2
SID=dense_barge-004
REF=$DL/runs/m2-lat1/ref-bmm

cd "$REPO"
mkdir -p $DL/runs/m2-lat1

run_variant () {
  NAME=$1; shift
  echo "===== variant: $NAME ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
      python -m m2.parity --ckpt $DL/runs/dense-pt1/p1_checkpoint.pt \
      --shards "$SHARDS" --scenarios "$SID" \
      --out $DL/runs/m2-lat1/$NAME "$@" \
      || echo "VARIANT FAILED: $NAME"
}

run_variant ref-bmm --experts batched_mm --save-picks $REF
run_variant eager --experts eager --ref-picks $REF/picks.json
run_variant eager-compile --experts eager --compile --ref-picks $REF/picks.json
echo "m2 latency bench done"
