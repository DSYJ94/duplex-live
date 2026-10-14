#!/bin/bash
#SBATCH --job-name=dl-lychee-clock
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/dl-lychee-clock-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
run_probe() {
  apptainer exec --bind /projects,/lus,/home \
    --env PYTHONPATH="$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$DL/repo" \
    --env HF_HUB_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 \
    "$DL/pytorch-new.sif" python -u -m evals.lychee_clock_probe "$@"
}
if run_probe; then
  echo 'ERROR: unaligned preprocessing unexpectedly passed the 25Hz clock assertion'
  exit 1
fi
run_probe --aligned
