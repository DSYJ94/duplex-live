#!/bin/bash
#SBATCH --job-name=dl-timed-reference-test
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/dl-timed-reference-test-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
SOURCE="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 8c6dfc101b7871baa428424bcdc583b74fb561d9
apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$SOURCE/moshi:$DL/repo" --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=2 --env NO_CUDA_GRAPH=1 --env NO_TORCH_COMPILE=1 \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-rag-7379661/bin/python" \
  -u -m unittest -v evals.harness.test_timed_reference_training
