#!/bin/bash
#SBATCH --job-name=dl-livebias
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/dl-livebias-%j.out

# Live T-gate on v11 under the repaired instruments + new stream
# contract (word continuation, sil-run end-of-turn), at two decode
# points: sil-bias 0 (contract effect alone vs gate #5, which ran
# pre-continuation) and sil-bias 4 (mid-dial: sweep showed 4 lifts
# exact word-onset hits 54->71/83 at ~+17% chatter on held-out
# lessons). Then the fixed contract trace.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

CKPT=$DL/runs/p1-v11/p1_checkpoint.pt
PROMPTS=$DL/runs/bakeoff/prompts
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif

cd "$REPO"

for CFG in "0 b0" "4 b4"; do
  set -- $CFG
  BIAS=$1; TAG=$2
  OUT=$DL/runs/harness-layera-p1-v11$TAG
  echo "===== live gate v11 sil-bias $BIAS -> $OUT ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      "$CONTAINER" \
      python -m evals.harness.run --live layera \
      --ckpt "$CKPT" --sil-bias "$BIAS" \
      --tests t1,t2,t3,t4,t5 --out "$OUT" --prompts "$PROMPTS"
done

echo "===== contract trace (v11) ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    "$CONTAINER" \
    python -m layera.contract_trace \
    --batch $DL/runs/synth-big7/audio --ckpt "$CKPT" \
    --shards $DL/runs/shards --classes timed_task --limit 3

echo "livebias done"
