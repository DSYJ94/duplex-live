#!/bin/bash
#SBATCH --job-name=dl-unseen
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=logs/dl-unseen-%j.out

# Unseen-wording gate: the gate prompts were ALREADY TTS voice (the
# tone-gap reading was wrong) — the real question is seen-script
# recall vs unseen-wording generalisation. Renders the same T-protocol
# shapes with new wording (no phrase-bank overlap) in the same user
# voice, then gates v3 (bias 0) and v11 (bias 4) on them.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif
PROMPTS=$DL/runs/bakeoff/prompts-unseen

cd "$REPO"

echo "===== render unseen prompts (minicpm TTS, user voice) ====="
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$DL/pylibs-minicpm:$DL/pylibs:$REPO \
    --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR "$CONTAINER" \
    python -m evals.render_unseen_prompts --out "$PROMPTS"

for CFG in "$DL/runs/p1-v3/p1_checkpoint.pt v3 0" \
           "$DL/runs/p1-v11/p1_checkpoint.pt v11b4 4"; do
  set -- $CFG
  CKPT=$1; TAG=$2; BIAS=$3
  OUT=$DL/runs/harness-layera-p1-$TAG-unseen
  echo "===== unseen gate $TAG (sil-bias $BIAS) -> $OUT ====="
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$DL/pylibs:$REPO \
      --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR "$CONTAINER" \
      python -m evals.harness.run --live layera \
      --ckpt "$CKPT" --sil-bias "$BIAS" \
      --tests t1,t2,t3,t4,t5 --out "$OUT" --prompts "$PROMPTS"
done
echo "unseen gate done"
