#!/bin/bash
#SBATCH --job-name=dl-sc1shard
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/dl-sc1shard-%j.out

# SCALING RUN stage 2 of 3: shard the scale1 corpus (encoder pass per
# lesson -> audio_feats). Consumes train_manifest.jsonl = ONLY lessons
# the quality filter passed. Resume-safe: prepare skips scenarios that
# already have a shard file, so a timeout costs time, not work. Chains
# stage 3 (training) afterok so the whole run is cert-lapse-proof.
# SUBMITTED BY evals/scale1-data.slurm (dependency on the filter job) —
# do not launch by hand unless the chain is broken.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

cd "$REPO"

apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -m layera.prepare --batch $DL/runs/synth-scale1/audio \
        --out $DL/runs/shards-scale1

SHARD_JOB=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID \
    -o $DL/logs/dl-sc1train-%j.out $DL/repo/evals/scale1-train.slurm)
echo "[scale1-shard] chained train job: $SHARD_JOB (afterok:$SLURM_JOB_ID)"
echo "scale1 shards built"
