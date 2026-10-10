#!/bin/bash
#SBATCH --job-name=dl-sq2shard
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/dl-sc1shard-%j.out

# EAR-CLOCK-FIXED SHARDING (2026-10-03 night): encode_audio now
# resamples tower tokens onto the tick grid (52->50 per 4s window,
# nearest-time gather). RE-ENCODES all 163 verified lessons (the
# shards-scale1-q feats carry the 4% skew). First a one-lesson
# alignment smoke, then the full pass, then the chained retrain. Resume-safe: prepare skips scenarios that
# already have a shard file, so a timeout costs time, not work. Chains
# stage 3 (training) afterok so the whole run is cert-lapse-proof.
# SUBMITTED BY evals/scale1-data.slurm (dependency on the filter job) —
# do not launch by hand unless the chain is broken.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs-new:$REPO

cd "$REPO"

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/duplex-live/pytorch-new.sif \
    python -m layera.prepare --batch $DL/runs/synth-scale1-q/audio \
        --out $DL/runs/shards-scale1-q2 --per-class 999

SHARD_JOB=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID \
    -o $DL/logs/dl-sq2train-%j.out $DL/repo/evals/scale1q2-train.slurm)
echo "[scale1-shard] chained train job: $SHARD_JOB (afterok:$SLURM_JOB_ID)"
echo "scale1 shards built"
