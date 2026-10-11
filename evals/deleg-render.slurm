#!/bin/bash
#SBATCH --job-name=dl-deleg2
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/dl-deleg2-%j.out

# DELEGATION + STREAMING corpus (2026-10-11). Timelines already written
# (job 7257640: 60 delegation + 40 timed_task). This renders them with
# ElevenLabs + the fail-closed judge, then self-chains filter ->
# timing+relabel -> stamp. Arm-B verdict closed the content path
# (WER 1.0-2.5, dialogs muted) — delegation behaviours are PLACEMENT-
# level skills (when to fire <delegate>, how to bridge waits, when
# staged results arrive), the class of thing that trains reliably.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp

source $DL/keys/elevenlabs.env
export PYTHONPATH=$DL/pylibs-new:$REPO

cd "$REPO"
OUT=$DL/runs/synth-deleg/audio

for k in 0 1; do
  apptainer exec --nv --bind /projects,/lus,/home \
      --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
      --env ELEVENLABS_API_KEY=$ELEVENLABS_API_KEY \
      /projects/u6si/duplex-live/pytorch-new.sif \
      python -m data.synthetic.render \
      --timelines $DL/runs/synth-deleg/timelines --out "$OUT" \
      --engine elevenlabs --clip-cache $DL/runs/clip-cache \
      --slice $k/2 --repair 3 --assets /tmp \
      > $DL/logs/render-deleg-w$k.log 2>&1 &
done
wait
cat $OUT/manifest.part*of2.jsonl >> $OUT/manifest.jsonl 2>/dev/null || true
rm -f $OUT/manifest.part*of2.jsonl
echo "[deleg] render done: $(wc -l < $OUT/manifest.jsonl) shipped"

F=$(sbatch --parsable --export=ALL,AUDIO=$OUT,CACHE=$DL/runs/clip-cache \
    -o $DL/logs/synth-filter-deleg-%j.out $REPO/synth-filter.slurm)
echo "[deleg] chained filter: $F"
T=$(sbatch --parsable --dependency=afterok:$F --export=ALL,AUDIO=$OUT \
    -o $DL/logs/dl-delegqt-%j.out $REPO/evals/scale1-timing.slurm)
echo "[deleg] chained timing+stamp: $T"
