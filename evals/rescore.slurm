#!/bin/bash
#SBATCH --job-name=dl-rescore
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --time=00:40:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/dl-rescore-%j.out

# Rescore every existing gate recording with the REPAIRED instruments
# (2026-09-22): end-anchored latencies, n_answered, t2-from-barge-end,
# t4-part2-end. CPU-only — replays wavs, no model. Overwrites
# harness_report.json in place (start-anchored history lives in
# EXPERIMENTS.md).

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO

CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif
DIRS=(
  runs/harness-layera-p1-v2-400step-unweighted
  runs/harness-layera-p1-v3
  runs/harness-layera-p1-v10
  runs/harness-layera-p1-v11
)

cd "$REPO"
for d in "${DIRS[@]}"; do
  if [ ! -d "$d" ]; then echo "skip $d (missing)"; continue; fi
  echo "=== rescore $d ==="
  apptainer exec --cleanenv --bind /projects,/lus,/home \
    --env TMPDIR=$TMPDIR --env PYTHONPATH=$PYTHONPATH "$CONTAINER" \
    python -m evals.harness.run --replay "$d" \
    > "$d/rescore_$(date +%Y%m%d).log" 2>&1 \
    || { echo "RESCORE FAILED for $d"; exit 1; }
  grep -E "^t[0-9]" "$d/rescore_$(date +%Y%m%d).log" | tail -6 || true
done

echo "=== gate regression test (CPU) ==="
apptainer exec --cleanenv --bind /projects,/lus,/home \
  --env TMPDIR=$TMPDIR --env PYTHONPATH=$PYTHONPATH "$CONTAINER" \
  python -m evals.harness.test_gate_regressions \
  > "$DL/runs/gate_regressions_$(date +%Y%m%d).log" 2>&1 \
  && tail -1 "$DL/runs/gate_regressions_$(date +%Y%m%d).log" \
  || { echo "REGRESSION TEST FAILED"; exit 1; }
echo "rescore done"
