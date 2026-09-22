#!/bin/bash
#SBATCH --job-name=dl-sweep
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/dl-sweep-%j.out

# Decoding sweep on the CURRENT best checkpoint (v11): is under-speaking
# substantially a DECODE decision? sil-bias subtracts from the <sil>
# logit at decode — no training. Free-run (self-history) on 4 held-out
# lessons per config, word continuation ON (the training contract).
# Reads: word-onset hits, false-speech ticks, t+30s thief, continuation.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

CKPT=${CKPT:-$DL/runs/p1-v11/p1_checkpoint.pt}
SHARDS=$DL/runs/shards
BATCH=$DL/runs/synth-big7/audio
export CKPT SHARDS BATCH   # the heredoc below runs in a child bash —
                           # unexported vars would arrive empty

cd "$REPO"
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    bash -s <<'EOF'
set -euo pipefail
for B in 0 1 2 4 8 16; do
  echo "===== sil-bias $B ====="
  python -m layera.freerun_diag \
    --batch "$BATCH" --ckpt "$CKPT" --shards "$SHARDS" \
    --classes timed_task,barge_in --limit 4 --max-ticks 900 \
    --sil-bias "$B" 2>&1 | grep -v Warning
done

echo "===== contract trace (v11, training-vs-live streams) ====="
python -m layera.contract_trace \
  --batch "$BATCH" --ckpt "$CKPT" --shards "$SHARDS" \
  --classes timed_task --limit 3
EOF
echo "sweep done"
