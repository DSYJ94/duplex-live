#!/usr/bin/env bash
# Heavy source decompression/verification stays inside a zero-GPU CPU job.
set -euo pipefail
exec /usr/bin/python3 -B -u -m training.personaplex.matched_cpu \
  --source "${1:?exact source SHA}" --out "${2:?new full input artifact}" \
  --tracker "${3:?new actual process tracker}" --require-slurm-cpu
